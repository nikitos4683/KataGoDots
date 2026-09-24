#!/usr/bin/env python3
"""Choose the selfplay model format from a training checkpoint's architecture."""

import json
import os
import sys

from katago.train.load_model import load_checkpoint


BIN_BLOCK_KINDS = frozenset({
    "regular", "regulargpool",
    "bottlenest2", "bottlenest2gpool", "bottlenest3", "bottlenest3gpool",
    "attnrope", "ffnsg",
    "bottlenest2transformerropesg", "bottlenest3transformerropesg",
})

ONNX_REQUIRED_OPTIONS = (
    "attention_qk_norm",
    "inline_registers",
    "attention_num_rw_registers",
    "transformer_ffn_depthwise_conv",
    "use_trunk_channel_gate",
    "use_trunk_residual_backout",
    "rmsnorm_spatial_cgroup_size",
)


def choose_format(config):
    # The binary exporter and C++ descriptor support the ordinary transformer
    # blocks, including the tf3 models. Other blocks and options require the
    # full PyTorch graph supplied by the ONNX exporter.
    if any(kind not in BIN_BLOCK_KINDS for _, kind in config["block_kind"]):
        return "onnx"
    if any(config.get(option) for option in ONNX_REQUIRED_OPTIONS):
        return "onnx"
    return "bin"


def main(checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint.get("config")
    if config is None:
        with open(os.path.join(os.path.dirname(checkpoint_path), "model.config.json"), encoding="utf-8") as src:
            config = json.load(src)
    print(choose_format(config))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: model_export_format.py CHECKPOINT")
    main(sys.argv[1])
