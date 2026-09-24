import copy

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
ort = pytest.importorskip("onnxruntime")

from export_model_onnx import RawHeadsForEngine, export_onnx
from katago.train.model_pytorch import Game, Model, TransformerAttentionBlock
from katago.train.modelconfigs import base_config_of_name


@pytest.mark.parametrize(
    "name",
    [
        "b5c48h3tfr",  # FFN without SwiGLU
        "b9c96h3tgabs",  # GAB
        "b5c384h6nbttflrtab2cheaps",  # frequency-mixing TAB
        "b6c384h6nbttflrs-qkn-ireg16",  # QK norm and inline registers
        "b6c384h6nbttflrs-dwc",  # depthwise FFN
    ],
)
def test_onnx_export_matches_rectangular_dots_model(tmp_path, monkeypatch, name):
    monkeypatch.setenv("KATAGO_FLEX_ATTENTION", "0")
    config = copy.deepcopy(base_config_of_name[name])
    config["block_kind"] = config["block_kind"][:1]
    model = Model(config, 8, 6, games=[Game.DOTS])
    model.initialize()
    if name in ("b5c48h3tfr", "b6c384h6nbttflrs-dwc"):
        model.use_flex_attention = True
    original_flex_setting = model.use_flex_attention

    path = tmp_path / f"{name}.onnx"
    export_onnx(model, str(path), name)
    assert model.use_flex_attention == original_flex_setting
    assert all(not hasattr(block, "onnx_export") for block in model.modules()
               if isinstance(block, TransformerAttentionBlock))
    model.use_flex_attention = False  # CPU parity evaluation uses the ordinary attention path.
    graph = onnx.load(str(path))
    metadata = {entry.key: entry.value for entry in graph.metadata_props}
    assert metadata["katago.metadataVersion"] == "1"
    assert metadata["katago.dotsGame"] == "true"
    assert metadata["katago.preferExcludeTerritoryAdjacentToAtari"] == "false"
    assert metadata["katago.build.nnXLen"] == "8"
    assert metadata["katago.build.nnYLen"] == "6"

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for smaller_board in (False, True):
        spatial = torch.randn(2, 22, 6, 8)
        global_input = torch.randn(2, 19, 1, 1)
        mask = torch.ones(2, 1, 6, 8)
        if smaller_board:
            mask[:, :, -1, :] = 0.0
            mask[:, :, :, -1] = 0.0
        spatial[:, 0] = mask[:, 0]
        with torch.no_grad():
            expected = RawHeadsForEngine(model).eval()(spatial, global_input, mask)

        actual = session.run(
            None,
            {
                "InputSpatial": spatial.numpy(),
                "InputGlobal": global_input.numpy(),
                "InputMask": mask.numpy(),
            },
        )
        for observed, reference in zip(actual, expected):
            np.testing.assert_allclose(observed, reference.numpy(), rtol=1e-4, atol=1e-5)
