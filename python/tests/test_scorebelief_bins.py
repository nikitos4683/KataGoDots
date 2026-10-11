"""Score-belief coordinates must match the C++ training writer's game-specific bins."""

import copy
import math

import pytest
import torch
from torch.optim.swa_utils import AveragedModel

from katago.train.load_model import load_model
from katago.train.metrics_pytorch import Metrics
from katago.train.model_pytorch import Game, Model
from katago.train.modelconfigs import base_config_of_name


@pytest.fixture(autouse=True)
def disable_flex_attention(monkeypatch):
    monkeypatch.setenv("KATAGO_FLEX_ATTENTION", "0")


@pytest.fixture(scope="module", autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_model(game, intermediate=False, x=4, y=3):
    config = copy.deepcopy(base_config_of_name["b5c48h3tfr"])
    config["block_kind"] = config["block_kind"][:1]
    if intermediate:
        config["has_intermediate_head"] = True
        config["intermediate_head_blocks"] = 1
    model = Model(config, pos_len_x=x, pos_len_y=y, games=[game])
    model.initialize()
    return model


@pytest.mark.parametrize("x,y", [(4, 3), (39, 32)])
@pytest.mark.parametrize("score", [0.0, -3.0, 4.0, -2.75, 3.25])
def test_dots_distribution_decodes_cpp_integer_bins(x, y, score):
    model = small_model(Game.DOTS, x=x, y=y)
    metrics = Metrics(1, model)
    head = model.value_head
    distribution = torch.zeros(head.scorebelief_len)
    # C++ splits fractional Dots scores between floor(score) and floor(score)+1.
    lower = math.floor(score)
    fraction = score - lower
    distribution[head.scorebelief_mid + lower] = 1.0 - fraction
    distribution[head.scorebelief_mid + lower + 1] = fraction
    expected_score = torch.sum(distribution * metrics.score_belief_offset_vector)
    assert expected_score.item() == pytest.approx(score)


@pytest.mark.parametrize("game,offset", [(Game.DOTS, 0.0), (Game.GO, 0.5)])
def test_score_coordinates_reach_main_and_intermediate_heads(game, offset):
    model = small_model(game, intermediate=True)
    model.eval()
    captured = {}

    def capture(name):
        def hook(module, inputs):
            captured[name] = inputs[0].detach().clone()
        return hook

    heads = {"main": model.value_head, "intermediate": model.intermediate_value_head}
    for name, head in heads.items():
        head.linear_s2off.register_forward_pre_hook(capture(name))
        assert head.score_belief_offset_vector[head.scorebelief_mid].item() == offset
        assert head.score_belief_offset_vector[head.scorebelief_mid - 1].item() == offset - 1.0

    spatial = torch.zeros(1, 22, 3, 4)
    spatial[:, 0] = 1.0
    with torch.no_grad():
        model(spatial, torch.zeros(1, 19))
    assert set(captured) == set(heads)
    for name, head in heads.items():
        assert captured[name][0, head.scorebelief_mid, 0].item() == pytest.approx(0.05 * offset)


def test_go_distribution_keeps_cpp_half_integer_bins():
    model = small_model(Game.GO)
    head = model.value_head
    distribution = torch.zeros(head.scorebelief_len)
    # Go score zero is a 50/50 split between the -0.5 and +0.5 buckets.
    distribution[head.scorebelief_mid - 1:head.scorebelief_mid + 1] = 0.5
    assert torch.sum(distribution * head.score_belief_offset_vector).item() == 0.0


def test_dots_checkpoint_rebuilds_score_coordinates_for_main_and_swa(tmp_path):
    model = small_model(Game.DOTS, intermediate=True)
    state = model.state_dict()
    swa_state = AveragedModel(model).state_dict()
    for name, head in [("value_head", model.value_head),
                       ("intermediate_value_head", model.intermediate_value_head)]:
        for buffer in ["score_belief_offset_vector", "score_belief_offset_bias_vector"]:
            assert f"{name}.{buffer}" not in state
        # Legacy checkpoints may contain the derived Go-style buffers. The
        # loader must ignore them and reconstruct the Dots coordinates instead.
        legacy_offsets = head.score_belief_offset_vector + 0.5
        state[f"{name}.score_belief_offset_vector"] = legacy_offsets
        state[f"{name}.score_belief_offset_bias_vector"] = 0.05 * legacy_offsets
        swa_state[f"module.{name}.score_belief_offset_vector"] = legacy_offsets
        swa_state[f"module.{name}.score_belief_offset_bias_vector"] = 0.05 * legacy_offsets
    checkpoint = tmp_path / "checkpoint.ckpt"
    torch.save(dict(config=model.config, model=state, swa_model=swa_state,
                    pos_len_x=4, pos_len_y=3, games=[Game.DOTS.name]), checkpoint)
    restored, swa_model, _ = load_model(checkpoint, use_swa=True, device="cpu")
    for loaded in [restored, swa_model]:
        assert (loaded.pos_len_x, loaded.pos_len_y) == (4, 3)
        for head in [loaded.value_head, loaded.intermediate_value_head]:
            assert head.score_belief_offset_vector[head.scorebelief_mid].item() == 0.0
            assert head.score_belief_offset_bias_vector[head.scorebelief_mid].item() == 0.0
