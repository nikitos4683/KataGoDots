import copy

import pytest
import torch

from katago.train.model_pytorch import Game, Model, TransformerAttentionBlock
from katago.train.modelconfigs import base_config_of_name


@pytest.mark.parametrize(
    "name,learnable_rope",
    [
        ("b5c48h3tfr", False),
        ("b5c48h3tfr", True),
        ("b5c384h6nbttfgabs", False),
        ("b5c384h6nbttftabs", False),
        ("b5c384h6nbttflrtab2cheaps", False),
        ("b6c384h6nbttflrs-qkn-ireg16", False),
        ("b6c384h6nbttflrs-dwc", False),
    ],
)
def test_dots_transformer_forward_on_rectangular_board(monkeypatch, name, learnable_rope):
    monkeypatch.setenv("KATAGO_FLEX_ATTENTION", "0")
    config = copy.deepcopy(base_config_of_name[name])
    config["block_kind"] = config["block_kind"][:1]
    if learnable_rope:
        config["learnable_rope"] = True

    model = Model(config, pos_len_x=8, pos_len_y=6, games=[Game.DOTS])
    model.initialize()
    model.eval()

    spatial = torch.zeros((1, 22, 6, 8))
    spatial[:, 0] = 1.0
    with torch.no_grad():
        outputs = model(spatial, torch.zeros((1, 19)))

    policy = outputs[0][0]
    assert policy.shape == (1, 6, 8 * 6 + 1)
    assert torch.isfinite(policy).all()


@pytest.mark.parametrize(
    "name",
    [
        "b5c384h6nbttfgabs",
        "b5c384h6nbttflrtab2cheaps",
        "b6c384h6nbttflrs-qkn-ireg16",
        "b6c384h6nbttflrs-dwc",
    ],
)
def test_dots_transformer_backward_on_rectangular_board(monkeypatch, name):
    monkeypatch.setenv("KATAGO_FLEX_ATTENTION", "0")
    config = copy.deepcopy(base_config_of_name[name])
    config["block_kind"] = config["block_kind"][:1]
    model = Model(config, pos_len_x=8, pos_len_y=6, games=[Game.DOTS])
    model.initialize()

    spatial = torch.zeros((2, 22, 6, 8))
    spatial[:, 0] = 1.0
    policy = model(spatial, torch.zeros((2, 19)))[0][0]
    policy.square().mean().backward()

    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires a CUDA GPU")
@pytest.mark.parametrize("name", [
    "b5c48h3tfr",
    "b6c384h6nbttflrs-dwc",
    "b6c384h6nbttflrs-qkn",
])
@pytest.mark.parametrize("smaller_board", [False, True])
def test_flex_attention_matches_export_path(monkeypatch, name, smaller_board):
    monkeypatch.setenv("KATAGO_FLEX_ATTENTION", "1")
    config = copy.deepcopy(base_config_of_name[name])
    config["block_kind"] = config["block_kind"][:1]
    model = Model(config, pos_len_x=8, pos_len_y=6, games=[Game.DOTS]).cuda()
    model.initialize()
    model.eval()
    assert model.use_flex_attention

    spatial = torch.randn((1, 22, 6, 8), device="cuda")
    spatial[:, 0] = 1.0
    if smaller_board:
        spatial[:, 0, -1, :] = 0.0
        spatial[:, 0, :, -1] = 0.0
    global_input = torch.randn((1, 19), device="cuda")
    with torch.no_grad():
        flex_outputs = model(spatial, global_input)[0]
        model.use_flex_attention = False
        for module in model.modules():
            if isinstance(module, TransformerAttentionBlock):
                module.onnx_export = True
        export_outputs = model(spatial, global_input)[0]

    for flex, exported in zip(flex_outputs[:5], export_outputs[:5]):
        torch.testing.assert_close(flex, exported, rtol=1e-5, atol=2e-5)
