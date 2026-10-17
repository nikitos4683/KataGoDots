# Reader of the exported sparse Dots model files (.bin / .bin.gz, model version 18), mirroring the C++ parser.
# Used to check the export round trip: the weights read back into a fresh model must reproduce the outputs
# that the engine uses.

import gzip
import struct
from typing import Dict, List, Tuple

import numpy as np
import torch

from ..train import modelconfigs
from ..train.model_pytorch import Game, NormMask, TransformerAttentionBlock, TransformerFFNBlock
from ..train.model_sparse_dots import NUM_CHAIN_CATEGORIES, NUM_POINT_FEATURES, CrossAttentionBlock, SparseDotsModel


class BinReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def _skip_whitespace(self):
        while self.pos < len(self.data) and self.data[self.pos] in b" \t\r\n":
            self.pos += 1

    def token(self) -> str:
        self._skip_whitespace()
        start = self.pos
        while self.pos < len(self.data) and self.data[self.pos] not in b" \t\r\n":
            self.pos += 1
        return self.data[start:self.pos].decode("ascii")

    def int(self) -> int:
        return int(self.token())

    def float(self) -> float:
        return float(self.token())

    def floats(self, n: int) -> np.ndarray:
        self._skip_whitespace()
        assert self.data[self.pos:self.pos + 5] == b"@BIN@", "Expected binary floats"
        self.pos += 5
        values = np.frombuffer(self.data, dtype="<f4", count=n, offset=self.pos).copy()
        self.pos += 4 * n
        return values

    def expect(self, value: str):
        token = self.token()
        assert token == value, f"Expected {value}, got {token}"

    def matmul(self, name: str) -> np.ndarray:
        """Returns the weight in torch layout [out, in]."""
        self.expect(name)
        c_in = self.int()
        c_out = self.int()
        return self.floats(c_in * c_out).reshape(c_in, c_out).T.copy()

    def matbias(self, name: str) -> np.ndarray:
        self.expect(name)
        return self.floats(self.int())

    def activation(self, name: str) -> str:
        self.expect(name)
        return self.token()

    def transformer_norm(self, name: str) -> np.ndarray:
        self.expect(name)
        c = self.int()
        self.float()
        return self.floats(c)

    def rmsnorm(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        self.expect(name)
        c = self.int()
        self.float()
        assert self.int() == 1, "Expected a spatial RMSNorm"
        assert self.int() == 0
        return self.floats(c), self.floats(c)

    def batchnorm(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the merged scale and bias (the norms of the sparse trunk are not batch norms)."""
        self.expect(name)
        c = self.int()
        epsilon = self.float()
        has_scale = self.int()
        has_bias = self.int()
        mean = self.floats(c)
        var = self.floats(c)
        scale = self.floats(c) if has_scale else np.ones(c, np.float32)
        bias = self.floats(c) if has_bias else np.zeros(c, np.float32)
        std = np.sqrt(var + epsilon)
        return scale / std, bias - mean * scale / std

    def conv1x1(self, name: str) -> np.ndarray:
        self.expect(name)
        assert self.int() == 1 and self.int() == 1, "Expected a 1x1 convolution"
        c_in = self.int()
        c_out = self.int()
        self.int()
        self.int()
        return self.floats(c_in * c_out).reshape(c_in, c_out).T.copy()

    def rope_freqs(self, name: str) -> np.ndarray:
        self.expect(name)
        h = self.int()
        p = self.int()
        two = self.int()
        return self.floats(h * p * two).reshape(h, p, two)


def _set(param: torch.Tensor, value: np.ndarray):
    with torch.no_grad():
        param.copy_(torch.from_numpy(np.ascontiguousarray(value)).view(param.shape))


def _load_norm(reader: BinReader, name: str, norm: NormMask):
    scale, bias = reader.batchnorm(name)
    assert not norm.is_using_batchnorm
    norm_scale = norm.scale if norm.scale is not None else 1.0
    if norm.gamma is not None:
        gamma = scale / norm_scale - (1.0 if norm.gamma_weight_decay_center_1 else 0.0)
        _set(norm.gamma, gamma)
    else:
        assert np.allclose(scale, norm_scale, rtol=1e-6), f"{name}: unexpected scale"
    _set(norm.beta, bias)


def _load_attention(reader: BinReader, name: str, block):
    reader.expect(name)
    assert reader.int() == block.num_heads and reader.int() == block.num_kv_heads
    assert reader.int() == block.q_head_dim and reader.int() == block.v_head_dim
    assert reader.int() == 1 and reader.int() == 1
    _set(block.norm1.weight, reader.transformer_norm(name + ".norm1"))
    _set(block.q_proj.weight, reader.matmul(name + ".q_proj"))
    _set(block.k_proj.weight, reader.matmul(name + ".k_proj"))
    _set(block.v_proj.weight, reader.matmul(name + ".v_proj"))
    _set(block.out_proj.weight, reader.matmul(name + ".out_proj"))
    _set(block.rope_freqs, reader.rope_freqs(name + ".rope_freqs"))


def _load_ffn(reader: BinReader, name: str, block: TransformerFFNBlock):
    reader.expect("transformer_ffn_block")
    reader.expect(name)
    assert reader.int() == block.c_main and reader.int() == block.ffn_dim and reader.int() == 1
    _set(block.norm.weight, reader.transformer_norm(name + ".norm"))
    _set(block.ffn_linear1.weight, reader.matmul(name + ".ffn_linear1"))
    _set(block.ffn_linear_gate.weight, reader.matmul(name + ".ffn_linear_gate"))
    _set(block.ffn_linear2.weight, reader.matmul(name + ".ffn_linear2"))


def _load_linear(reader: BinReader, name: str, linear: torch.nn.Linear):
    _set(linear.weight, reader.matmul(name))
    if linear.bias is not None:
        _set(linear.bias, reader.matbias(name + ".bias"))


def load_sparse_dots_bin(path: str, config: modelconfigs.ModelConfig) -> Tuple[SparseDotsModel, Dict]:
    """Reads the file into a fresh model of the config. Only the outputs exported for the engine are restored:
    policy outputs 0 and 5 (stored at 0 and 5), the value, the misc values used by the engine and ownership."""
    with open(path, "rb") as f:
        data = f.read()
    if path.endswith(".gz"):
        data = gzip.decompress(data)
    reader = BinReader(data)
    info = {"name": reader.token()}
    header = reader.token().split(",")
    info["version"] = int(header[0])
    pos_len_x = int(header[1])
    pos_len_y = int(header[2])
    assert header[3] == "DOTS"
    assert info["version"] == modelconfigs.SPARSE_DOTS_VERSION
    model = SparseDotsModel(config, pos_len_x, pos_len_y, [Game.DOTS])
    model.initialize()

    assert reader.int() == NUM_POINT_FEATURES and reader.int() == 8
    info["multipliers"] = [reader.float() for _ in range(7)]
    meta = [reader.int() for _ in range(8)]
    assert meta == [0] * 8
    reader.expect("sparse_dots_model")
    assert [reader.int() for _ in range(2)] == [8, 1]
    assert reader.int() == NUM_CHAIN_CATEGORIES
    assert [reader.int() for _ in range(5)] == [0] * 5

    _load_linear(reader, "model.linear_point", model.linear_point)
    _set(model.global_token, reader.matbias("model.global_token"))
    _set(model.linear_global.weight, reader.matmul("model.linear_global"))

    reader.expect("trunk")
    assert reader.int() == len(model.blocks)
    assert reader.int() == model.c_trunk and reader.int() == model.c_mid
    for _ in range(3):
        reader.int()
    assert reader.int() == 1
    assert [reader.int() for _ in range(5)] == [0] * 5
    for i, block in enumerate(model.blocks):
        name = f"model.blocks.{i}"
        reader.expect("nested_bottleneck_block")
        reader.expect(name)
        assert reader.int() == len(block.blockstack)
        _load_norm(reader, name + ".normactconvp.norm", block.normactconvp.norm)
        reader.activation(name + ".normactconvp.act")
        _set(block.normactconvp.conv.weight, reader.conv1x1(name + ".normactconvp.conv"))
        for j, sub in enumerate(block.blockstack):
            subname = f"{name}.blockstack.{j}"
            if isinstance(sub, TransformerAttentionBlock):
                reader.expect("transformer_attention_block")
                _load_attention(reader, subname, sub)
            else:
                _load_ffn(reader, subname, sub)
        _load_norm(reader, name + ".normactconvq.norm", block.normactconvq.norm)
        reader.activation(name + ".normactconvq.act")
        _set(block.normactconvq.conv.weight, reader.conv1x1(name + ".normactconvq.conv"))
    gamma, beta = reader.rmsnorm("model.norm_trunkfinal")
    _set(model.norm_trunkfinal.gamma, gamma)
    _set(model.norm_trunkfinal.beta, beta)
    reader.activation("model.act_trunkfinal")
    reader.expect("model.chain_bias")
    blocks = model.trunk_attention_blocks()
    shape = (reader.int(), reader.int(), reader.int())
    assert shape == (len(blocks), blocks[0].num_heads, NUM_CHAIN_CATEGORIES)
    tables = reader.floats(shape[0] * shape[1] * shape[2]).reshape(shape)
    for block, table in zip(blocks, tables):
        _set(block.chain_bias, table)

    decoder = model.decoder
    reader.expect("model.decoder")
    assert reader.int() == decoder.c_dec and reader.int() == decoder.num_layers
    assert [reader.int() for _ in range(2)] == [0, 0]
    _load_linear(reader, "model.decoder.linear_cand", decoder.linear_cand)
    _load_linear(reader, "model.decoder.linear_ground", decoder.linear_ground)
    _set(decoder.linear_pool.weight, reader.matmul("model.decoder.linear_pool"))
    for i in range(decoder.num_layers):
        reader.expect("transformer_cross_attention_block")
        _load_attention(reader, f"model.decoder.layers.{2*i}", decoder.layers[2 * i])
        _load_ffn(reader, f"model.decoder.layers.{2*i+1}", decoder.layers[2 * i + 1])
    _set(decoder.norm_final.weight, reader.transformer_norm("model.decoder.norm_final"))
    out = reader.matmul("model.decoder.linear_out")
    weight = np.zeros(tuple(decoder.linear_out.weight.shape), np.float32)
    weight[0] = out[0]
    weight[5] = out[1]
    _set(decoder.linear_out.weight, weight)

    vh = model.value_head
    reader.expect("model.value_head")
    assert [reader.int() for _ in range(3)] == [0, 0, 0]
    _set(vh.linear1.weight, reader.matmul("model.value_head.linear1"))
    _set(vh.bias1, reader.matbias("model.value_head.bias1"))
    reader.activation("model.value_head.act1")
    _set(vh.linear2.weight, reader.matmul("model.value_head.linear2"))
    _set(vh.linear2.bias, reader.matbias("model.value_head.bias2"))
    reader.activation("model.value_head.act2")
    _set(vh.linear_valuehead.weight, reader.matmul("model.value_head.linear_valuehead"))
    _set(vh.linear_valuehead.bias, reader.matbias("model.value_head.bias_valuehead"))
    w = reader.matmul("model.value_head.linear_miscvaluehead")
    b = reader.matbias("model.value_head.bias_miscvaluehead")
    with torch.no_grad():
        vh.linear_miscvaluehead.weight[0:4] = torch.from_numpy(w[0:4])
        vh.linear_miscvaluehead.bias[0:4] = torch.from_numpy(b[0:4])
        vh.linear_moremiscvaluehead.weight[0:2] = torch.from_numpy(w[4:6])
        vh.linear_moremiscvaluehead.bias[0:2] = torch.from_numpy(b[4:6])
    _set(vh.linear_ownership.weight, reader.matmul("model.value_head.linear_ownership"))

    reader._skip_whitespace()
    assert reader.pos == len(reader.data), "Trailing data in the model file"
    return model, info
