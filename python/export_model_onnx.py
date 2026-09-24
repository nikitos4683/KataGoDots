#!/usr/bin/env python3
"""Export a KataGoDots checkpoint as a self-contained TensorRT-compatible ONNX model."""

import argparse
import copy
import datetime
import json
import os
import re
import sys

import onnx
import torch

from katago.train import modelconfigs
from katago.train.load_model import load_model
from katago.train.model_pytorch import Game, Model, TransformerAttentionBlock


class RawHeadsForEngine(torch.nn.Module):
    """Expose the raw heads consumed by the C++ neural-net postprocessor."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        num_policy_outputs = model.policy_head.conv2p.weight.shape[0]
        if num_policy_outputs == 6:
            self.policy_indices = [0, 5]
        elif num_policy_outputs == 8:
            self.policy_indices = [0, 5, 6, 7]
        else:
            raise ValueError(f"Unsupported policy head with {num_policy_outputs} outputs")

    def forward(self, spatial, global_input, mask):
        # The engine supplies InputMask separately; channel 0 must agree with it.
        spatial = torch.cat((mask, spatial[:, 1:]), dim=1)
        outputs = self.model(spatial, global_input.flatten(1))[0]
        policy, value, misc, more_misc, ownership = outputs[:5]
        selected_policy = policy[:, self.policy_indices, :]
        batch = spatial.shape[0]
        height = spatial.shape[2]
        width = spatial.shape[3]
        return (
            selected_policy[:, :, -1:].reshape(batch, len(self.policy_indices), 1, 1),
            selected_policy[:, :, :-1].reshape(batch, len(self.policy_indices), height, width),
            value.reshape(batch, 3, 1, 1),
            torch.cat((misc[:, :4], more_misc[:, :2]), dim=1).reshape(batch, 6, 1, 1),
            ownership,
        )


def model_metadata(model, name):
    config = model.config
    if len(model.games) != 1:
        raise ValueError("ONNX inference supports one game per model")
    is_dots = model.games[0] == Game.DOTS
    policy_channels = 2 if model.policy_head.conv2p.weight.shape[0] == 6 else 4
    has_transformer = any("transformer" in kind or kind.startswith("attn") or kind.startswith("ffn")
                          for _, kind in config["block_kind"])
    has_nested = any("nest" in kind for _, kind in config["block_kind"])
    props = {
        "katago.metadataVersion": "1",
        "katago.name": name,
        "katago.modelVersion": str(config["version"]),
        "katago.numInputChannels": str(model.bin_input_shape[0]),
        "katago.numInputGlobalChannels": str(model.global_input_shape[0]),
        "katago.numInputMetaChannels": "0",
        "katago.numPolicyChannels": str(policy_channels),
        "katago.numValueChannels": "3",
        "katago.numScoreValueChannels": "6",
        "katago.numOwnershipChannels": "1",
        "katago.metaEncoderVersion": "0",
        "katago.preferPassAliveUnderSuicideRules": str(bool(config.get("always_compute_pass_alive_under_suicide_rules"))).lower(),
        "katago.preferExcludeTerritoryAdjacentToAtari": str(bool(config.get("exclude_territory_adjacent_to_atari"))).lower(),
        "katago.postProcess.tdScoreMultiplier": str(model.td_score_multiplier),
        "katago.postProcess.scoreMeanMultiplier": str(model.scoremean_multiplier),
        "katago.postProcess.scoreStdevMultiplier": str(model.scorestdev_multiplier),
        "katago.postProcess.leadMultiplier": str(model.lead_multiplier),
        "katago.postProcess.varianceTimeMultiplier": str(model.variance_time_multiplier),
        "katago.postProcess.shorttermValueErrorMultiplier": str(model.shortterm_value_error_multiplier),
        "katago.postProcess.shorttermScoreErrorMultiplier": str(model.shortterm_score_error_multiplier),
        "katago.build.nnXLen": str(model.pos_len_x),
        "katago.build.nnYLen": str(model.pos_len_y),
        "katago.build.requireExactNNLen": "false",
        "katago.build.transformerNHWC": "false",
        "katago.build.scale8Applied": "false",
        "katago.info.arch.trunkSpatialConvDepth": "0",
        "katago.info.arch.numParameters": str(sum(p.numel() for p in model.parameters())),
        "katago.info.arch.hasAnyTransformerBlocks": str(has_transformer).lower(),
        "katago.info.arch.hasAnyNestedBottleneckBlocks": str(has_nested).lower(),
    }
    if is_dots:
        props["katago.dotsGame"] = "true"
    return props


def export_onnx(model, output_path, name):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", name):
        raise ValueError("Model name must contain 1 to 96 letters, digits, underscores, or hyphens")
    if model.metadata_encoder is not None:
        raise ValueError("Metadata encoder models are not in the architecture registry")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    was_training = model.training
    old_flex_attention = model.use_flex_attention
    attention_blocks = [m for m in model.modules() if isinstance(m, TransformerAttentionBlock)]
    old_export_flags = [getattr(m, "onnx_export", None) for m in attention_blocks]
    try:
        model.eval()
        model.use_flex_attention = False
        for module in attention_blocks:
            module.onnx_export = True
        wrapper = RawHeadsForEngine(model).eval()
        x, y = model.pos_len_x, model.pos_len_y
        spatial = torch.zeros((2, model.bin_input_shape[0], y, x), dtype=torch.float32)
        spatial[:, 0] = 1.0
        global_input = torch.zeros((2, model.global_input_shape[0], 1, 1), dtype=torch.float32)
        mask = torch.ones((2, 1, y, x), dtype=torch.float32)
        batch = torch.export.Dim("batch", min=1)
        with torch.no_grad():
            torch.onnx.export(
                wrapper, (spatial, global_input, mask), output_path,
                input_names=["InputSpatial", "InputGlobal", "InputMask"],
                output_names=["OutputPolicyPass", "OutputPolicy", "OutputValue",
                              "OutputScoreValue", "OutputOwnership"],
                dynamic_shapes={"spatial": {0: batch}, "global_input": {0: batch}, "mask": {0: batch}},
                opset_version=20, dynamo=True, external_data=False,
            )

        graph = onnx.load(output_path)
        onnx.helper.set_model_props(graph, model_metadata(model, name))
        onnx.checker.check_model(graph)
        onnx.save(graph, output_path, save_as_external_data=False)
    finally:
        model.use_flex_attention = old_flex_attention
        for module, old_flag in zip(attention_blocks, old_export_flags):
            if old_flag is None:
                del module.onnx_export
            else:
                module.onnx_export = old_flag
        model.train(was_training)
    print(f"Exported ONNX model to {output_path}")


def write_selfplay_sidecars(output_path, other_state_dict):
    export_dir = os.path.dirname(os.path.abspath(output_path))
    train_state = other_state_dict.get("train_state", {})
    data = {}
    for key in ("global_step_samples", "total_num_data_rows"):
        if key in train_state:
            data[key] = train_state[key]
    if "running_metrics" in other_state_dict:
        data["extra_stats"] = {
            section: {
                key: value for key, value in values.items()
                if "sopt" not in key and "lopt" not in key
            }
            for section, values in other_state_dict["running_metrics"].items()
        }
        if "last_val_metrics" in other_state_dict:
            data["extra_stats"]["last_val_metrics"] = {
                section: {
                    key: value for key, value in values.items()
                    if "sopt" not in key and "lopt" not in key
                }
                for section, values in other_state_dict["last_val_metrics"].items()
                if section in ("sums", "weights")
            }
    with open(os.path.join(export_dir, "metadata.json"), "w", encoding="utf-8") as dst:
        json.dump(data, dst)
    with open(os.path.join(export_dir, "log.txt"), "w", encoding="utf-8") as dst:
        dst.write(f"Exported ONNX model {output_path} at {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-checkpoint")
    source.add_argument("-export-random-initialized-model")
    parser.add_argument("-out", required=True)
    parser.add_argument("-model-name", required=True)
    parser.add_argument("-use-swa", action="store_true")
    parser.add_argument("-selfplay-sidecars", action="store_true",
                        help="Write metadata.json and log.txt beside the model for selfplay uploads")
    parser.add_argument("-game", choices=["go", "dots"],
                        help="Override the checkpoint game, or choose the game for a random model")
    parser.add_argument("-pos-len-x", type=int)
    parser.add_argument("-pos-len-y", type=int)
    args = parser.parse_args()

    os.environ["KATAGO_FLEX_ATTENTION"] = "0"
    other_state_dict = {}
    if args.checkpoint:
        default_x = 39 if args.game == "dots" else 19
        default_y = 32 if args.game == "dots" else 19
        model, swa_model, other_state_dict = load_model(
            args.checkpoint, args.use_swa, "cpu",
            pos_len_x=args.pos_len_x or default_x,
            pos_len_y=args.pos_len_y or default_y,
        )
        model = swa_model if args.use_swa else model
        if args.game:
            model.games = [Game[args.game.upper()]]
    else:
        if args.use_swa:
            parser.error("-use-swa requires -checkpoint")
        config = copy.deepcopy(modelconfigs.config_of_name[args.export_random_initialized_model])
        game = args.game or "dots"
        x = args.pos_len_x or (39 if game == "dots" else 19)
        y = args.pos_len_y or (32 if game == "dots" else 19)
        model = Model(config, x, y, games=[Game[game.upper()]])
        model.initialize()
    export_onnx(model, args.out, args.model_name)
    if args.selfplay_sidecars:
        write_selfplay_sidecars(args.out, other_state_dict)


if __name__ == "__main__":
    main()
