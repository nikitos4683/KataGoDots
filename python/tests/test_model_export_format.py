from collections import Counter

import pytest

from katago.train.modelconfigs import base_config_of_name, config_of_name
from model_export_format import choose_format


@pytest.mark.parametrize(
    "name,expected",
    [
        ("b6c96", "bin"),
        ("b40c768nbt", "bin"),
        ("b20c384bt", "onnx"),
        ("b18c384dnbt1", "onnx"),
        ("b5c48h3tfr", "onnx"),
        ("b7c96h3tfrs", "bin"),
        ("b10c512h8nbt3tflrs", "bin"),
        ("b11c768h12nbt3tflrs", "bin"),
        ("b9c96h3tgabs", "onnx"),
        ("b5c384h6nbttflrtab2cheaps", "onnx"),
        ("b6c384h6nbttflrs-qkn-ireg16", "onnx"),
        ("b6c384h6nbttflrs-dwc", "onnx"),
        ("b2b10c96h3tfrs", "bin"),
    ],
)
def test_selfplay_export_format(name, expected):
    assert choose_format(base_config_of_name[name]) == expected


def test_every_base_architecture_has_an_export_path():
    formats = Counter(choose_format(config) for config in base_config_of_name.values())
    assert formats == {"bin": 50, "onnx": 18}


@pytest.mark.parametrize(
    "name",
    [
        "b10c384h6nbttflrs-fson-silu-rsnh",
        "b10c512h8nbt3tflrs-fson-silu-rsnh",
        "b11c768h12nbt3tflrs-fson-silu",
    ],
)
def test_official_transformer_configs_keep_binary_export(name):
    assert choose_format(config_of_name[name]) == "bin"
