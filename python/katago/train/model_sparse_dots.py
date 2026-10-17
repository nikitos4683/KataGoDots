# Sparse Dots model (model version 18), see docs/DotsSparseNN.md.
#
# A position is a set of tokens instead of a board grid:
#   GLOBAL (one per position, no position), POINT (placed dots outside the area interiors).
# The completed enclosures (areas) are not tokens: their facts are features of their boundary dots.
# Action candidates (reasonable points and grounding) are queries of a decoder with cross-attention
# to the trunk tokens. The trunk is the usual stack of nested bottleneck transformer blocks running
# on the token sequence ("1 x S board" with a mask).
#
# All the geometry (scaled coordinates, distances to the edges, RoPE) is computed
# here from the integer coordinates, with the same formulas as the C++ backends.

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn
import torch.nn.functional
from torch.amp import autocast

from ..train import modelconfigs
from ..train.model_pytorch import (
    EXTRA_SCORE_DISTR_RADIUS,
    ExtraOutputs,
    Game,
    NestedBottleneckTransformerBlock,
    SoftPlusWithGradientFloorFunction,
    TransformerAttentionBlock,
    TransformerFFNBlock,
    act,
    apply_learnable_rotary_emb,
    init_weights,
)

# Input feature layout -------------------------------------------------------------------------------------------

NUM_GLOBAL_FEATURES = 8
NUM_MOVES_AGO = 5
# owner, 4 edge distances, grounded, moves ago one-hot, then the areas whose boundary has the dot:
# their number, the dots inside them (mine, opponent's), the freshest moves ago one-hot
NUM_POINT_FEATURES = 19
NUM_POINT_DATA = 7  # pointDataNP: owner, grounded, movesAgo, areaCount, areaMovesAgo, insideMy, insideOpp
NUM_CAND_FEATURES = 8  # 4 edge distances, gainMy, gainOpp, inMyHouse, inOppHouse
NUM_GROUND_FEATURES = 1  # ground result
NUM_POLICY_OUTPUTS = 6
NUM_FUTUREPOS_OUTPUTS = 2

COORD_SCALE = 1.0 / 40.0
COUNT_SCALE = 1.0 / 20.0

# Initial learnable RoPE frequencies, rad per point along x and y. Up to pi (instead of 1 as in the dense model)
# so that from the start a head can tell a point from its neighbor: without a grid, the exact neighborhood of
# a dot or a candidate is seen only through RoPE.
ROPE_FREQ_INIT_MIN = 1.0 / 50.0
ROPE_FREQ_INIT_MAX = math.pi


def init_rope_freqs(num_heads: int, num_pairs: int) -> torch.Tensor:
    """Log-uniform magnitudes in [ROPE_FREQ_INIT_MIN, ROPE_FREQ_INIT_MAX] with random signs, [num_heads, num_pairs, 2]."""
    log_magnitudes = torch.empty(num_heads, num_pairs, 2).uniform_(math.log(ROPE_FREQ_INIT_MIN), math.log(ROPE_FREQ_INIT_MAX))
    magnitudes = torch.exp(log_magnitudes)
    signs = (torch.randint(0, 2, (num_heads, num_pairs, 2)) * 2 - 1).float()
    return magnitudes * signs

# Keys of the batch dict (training data arrays, docs/DotsSparseNN.md 7.2)
INPUT_KEYS = [
    "boardSizeN", "globalInputNC",
    "pointCountN", "pointCoordsNP", "pointDataNP",
    "candCountN", "candCoordsNM", "candDataNM",
    "groundN",
]
# Padded dimension (1-based axis) of each per-entity array
PADDED_ARRAYS = {
    "pointCoordsNP": "P", "pointDataNP": "P", "pointOwnershipTargetsNP": "P",
    "candCoordsNM": "M", "candDataNM": "M", "candPolicyTargetsNM": "M", "candFutureposTargetsNM": "M",
}
COUNT_KEY_OF_DIM = {"P": "pointCountN", "M": "candCountN"}

SPARSE_POSITIONS = "sparse_positions"


def is_sparse_dots_config(config: modelconfigs.ModelConfig) -> bool:
    return bool(config.get("sparse_dots", False))


def edge_features(coords: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
    """coords [..., 2] (x, y), board_size broadcastable [..., 2] (x_size, y_size). Returns [..., 4]."""
    x = coords[..., 0]
    y = coords[..., 1]
    w = board_size[..., 0]
    h = board_size[..., 1]
    return torch.stack([x * COORD_SCALE, y * COORD_SCALE, (w - 1.0 - x) * COORD_SCALE, (h - 1.0 - y) * COORD_SCALE], dim=-1)


def moves_ago_one_hot(moves_ago: torch.Tensor) -> torch.Tensor:
    """moves_ago [...] integer 0..5, 0 = none. Returns [..., 5]."""
    values = torch.arange(1, NUM_MOVES_AGO + 1, device=moves_ago.device)
    return (moves_ago.unsqueeze(-1) == values).to(torch.float32)


def point_features(point_coords: torch.Tensor, point_data: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
    """point_coords [N,P,2], point_data [N,P,7] (owner, grounded, movesAgo, areaCount, areaMovesAgo, insideMy,
    insideOpp), board_size [N,2]. Returns [N,P,19]."""
    coords = point_coords.to(torch.float32)
    data = point_data.to(torch.float32)
    return torch.cat([
        data[..., 0:1],
        edge_features(coords, board_size.unsqueeze(1)),
        data[..., 1:2],
        moves_ago_one_hot(point_data[..., 2].long()),
        data[..., 3:4],
        data[..., 5:6] * COUNT_SCALE,
        data[..., 6:7] * COUNT_SCALE,
        moves_ago_one_hot(point_data[..., 4].long()),
    ], dim=-1)


def cand_features(cand_coords: torch.Tensor, cand_data: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
    """cand_coords [N,M,2], cand_data [N,M,4] (gainMy, gainOpp, inMyHouse, inOppHouse). Returns [N,M,8]."""
    coords = cand_coords.to(torch.float32)
    data = cand_data.to(torch.float32)
    return torch.cat([
        edge_features(coords, board_size.unsqueeze(1)),
        data[..., 0:1] * COUNT_SCALE,
        data[..., 1:2] * COUNT_SCALE,
        data[..., 2:3],
        data[..., 3:4],
    ], dim=-1)


def ground_features(ground: torch.Tensor) -> torch.Tensor:
    """ground [N,2] (allowed, result). Returns [N,1,1]."""
    return (ground[:, 1:2].to(torch.float32) * COUNT_SCALE).unsqueeze(1)


def count_mask(counts: torch.Tensor, length: int) -> torch.Tensor:
    """counts [N]. Returns float mask [N, length]."""
    return (torch.arange(length, device=counts.device).unsqueeze(0) < counts.view(-1, 1).long()).to(torch.float32)


# Batch helpers ---------------------------------------------------------------------------------------------------

def trim_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Cuts the padding of the per-entity arrays down to the max count in the batch (at least 1)."""
    result = dict(batch)
    lengths = {}
    for dim, count_key in COUNT_KEY_OF_DIM.items():
        if count_key in batch:
            lengths[dim] = max(1, int(batch[count_key].max().item())) if batch[count_key].numel() > 0 else 1
    for key, dim in PADDED_ARRAYS.items():
        if key in result and dim in lengths:
            result[key] = result[key][:, :lengths[dim]]
    return result


def apply_symmetry_to_batch(batch: Dict[str, torch.Tensor], symmetry: int) -> Dict[str, torch.Tensor]:
    """Applies a board symmetry (same bits as C++ SymmetryHelpers: 1 flipY, 2 flipX, 4 transpose) to every
    coordinate of the batch. Transposing requires square boards in the whole batch.
    Outputs are indexed by tokens and candidates, so no inverse transformation is ever needed."""
    if symmetry == 0:
        return batch
    flip_y = (symmetry & 1) != 0
    flip_x = (symmetry & 2) != 0
    transpose = (symmetry & 4) != 0
    board_size = batch["boardSizeN"]
    if transpose:
        assert bool(torch.all(board_size[:, 0] == board_size[:, 1])), "Transposing symmetries need square boards"
    result = dict(batch)
    w = board_size[:, 0].view(-1, 1)
    h = board_size[:, 1].view(-1, 1)
    for key in ["pointCoordsNP", "candCoordsNM"]:
        coords = batch[key]
        x = coords[..., 0]
        y = coords[..., 1]
        if flip_x:
            x = (w - 1 - x).to(coords.dtype)
        if flip_y:
            y = (h - 1 - y).to(coords.dtype)
        if transpose:
            x, y = y, x
        result[key] = torch.stack([x, y], dim=-1)
    return result


# Positions for RoPE ----------------------------------------------------------------------------------------------

class SparsePositions:
    """Positions of the tokens and of the candidates, shared by all attention layers.
    Token order: [GLOBAL][POINT x P]. A point has its position. The GLOBAL token and the grounding candidate
    have no positions: their cos/sin tables are zero, so their rotated q/k vanish (NoPE)."""

    def __init__(self, point_xy, cand_xy):
        self.point_xy = point_xy  # [N,P,2] float
        self.cand_xy = cand_xy  # [N,M,2] float

    @staticmethod
    def _angles(xy, freqs):
        # xy [..., 2], freqs [H, Pp, 2] -> [..., H, Pp]
        return xy[..., 0, None, None] * freqs[:, :, 0] + xy[..., 1, None, None] * freqs[:, :, 1]

    def token_tables(self, freqs: torch.Tensor):
        """Returns cos, sin of shape [N, S, H, Pp] for the trunk tokens."""
        freqs = freqs.to(torch.float32)
        n = self.point_xy.shape[0]
        point_angles = self._angles(self.point_xy, freqs)
        global_zeros = point_angles.new_zeros((n, 1, freqs.shape[0], freqs.shape[1]))
        return torch.cat([global_zeros, torch.cos(point_angles)], dim=1), torch.cat([global_zeros, torch.sin(point_angles)], dim=1)

    def cand_tables(self, freqs: torch.Tensor):
        """Returns cos, sin of shape [N, M+1, H, Pp] for the candidates, the last one is the grounding."""
        freqs = freqs.to(torch.float32)
        angles = self._angles(self.cand_xy, freqs)
        n = self.cand_xy.shape[0]
        zeros = angles.new_zeros((n, 1, freqs.shape[0], freqs.shape[1]))
        return torch.cat([torch.cos(angles), zeros], dim=1), torch.cat([torch.sin(angles), zeros], dim=1)


def _expand_kv_tables(cos_k, sin_k, n_rep, num_heads):
    if n_rep == 1:
        return cos_k, sin_k
    cos_q = cos_k.unsqueeze(-2).expand(*cos_k.shape[:-1], n_rep, cos_k.shape[-1]).reshape(*cos_k.shape[:-2], num_heads, -1)
    sin_q = sin_k.unsqueeze(-2).expand(*sin_k.shape[:-1], n_rep, sin_k.shape[-1]).reshape(*sin_k.shape[:-2], num_heads, -1)
    return cos_q, sin_q


def _key_padding_bias(mask: torch.Tensor, dtype) -> torch.Tensor:
    """mask [N,S,1] or [N,S] -> additive bias [N,1,1,S]."""
    n = mask.shape[0]
    flat = mask.reshape(n, 1, 1, -1)
    bias = torch.zeros(flat.shape, dtype=dtype, device=mask.device)
    return bias.masked_fill(flat == 0, float("-inf"))


class SparseRoPEAttentionBlock(TransformerAttentionBlock):
    """Self-attention over the token sequence (NSC layout) with RoPE from the weighted position sets.
    Same parameters as TransformerAttentionBlock, so the export and the C++ descriptors are reused.
    Instances are created as TransformerAttentionBlock and switched to this class."""

    def forward(self, x, mask, mask_sum_hw, mask_sum: float, extra_outputs: Optional[ExtraOutputs], block_shared_data: Optional[Dict[str, Any]] = None):
        assert x.dim() == 3
        batch_size, seq_len, _ = x.shape
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        k = self.k_proj(x_norm).view(batch_size, seq_len, self.num_kv_heads, self.q_head_dim)
        v = self.v_proj(x_norm).view(batch_size, seq_len, self.num_kv_heads, self.v_head_dim)

        positions: SparsePositions = block_shared_data[SPARSE_POSITIONS]
        cos_k, sin_k = positions.token_tables(self.rope_freqs)
        cos_q, sin_q = _expand_kv_tables(cos_k, sin_k, self.n_rep, self.num_heads)
        q, k = apply_learnable_rotary_emb(q, k, cos_q, sin_q, cos_k, sin_k)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=_key_padding_bias(mask, q.dtype), dropout_p=0.0, scale=1.0 / math.sqrt(self.q_head_dim),
        )
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        result = self.out_proj(attn_output)
        if extra_outputs is not None:
            extra_outputs.report(self.name + ".out", result)
        return result


class SparseRMSNormMask(torch.nn.Module):
    """Spatial RMSNorm of the trunk tip over the valid tokens (NSC layout): one RMS per sample over all
    channels and tokens. Same parameters and math as RMSNormMask(spatial=True) of the dense model."""

    def __init__(self, c_in):
        super().__init__()
        self.c_in = c_in
        self.spatial = True
        self.cgroup_size = None
        self.norm = None
        self.eps = 1e-6
        self.gamma = torch.nn.Parameter(torch.ones(c_in))
        self.beta = torch.nn.Parameter(torch.zeros(c_in))

    def set_scale(self, scale):
        pass

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        reg_dict["output"].append(self.gamma)
        reg_dict["output"].append(self.beta)

    def forward(self, x, mask, num_tokens):
        """x [N,S,C], mask [N,S,1], num_tokens [N]."""
        mean_sq = torch.sum(x * x * mask, dim=(1, 2), keepdim=True) / (self.c_in * num_tokens.view(-1, 1, 1) + self.eps)
        rms = torch.sqrt(mean_sq + self.eps)
        return (x / rms * self.gamma.view(1, 1, -1) + self.beta.view(1, 1, -1)) * mask


def masked_mean_pool(x, mask, num_tokens):
    """x [N,S,C] (zero at padding), returns [N,C]."""
    return torch.sum(x * mask, dim=1) / num_tokens.view(-1, 1)


def board_area_offset(board_size):
    """sqrt(x_size * y_size) - 14, the board size term of KataGo's pooling. board_size [N,2] -> [N,1]."""
    return (torch.sqrt(board_size[:, 0] * board_size[:, 1]) - 14.0).view(-1, 1)


class CrossAttentionBlock(torch.nn.Module):
    """Attention of the candidates (queries) to the trunk tokens (keys and values) with learnable RoPE.
    RMSNorm -> projections -> RoPE -> attention -> output projection. Returns the residual."""

    def __init__(self, name, c_query, c_kv, num_heads, head_dim):
        super().__init__()
        self.name = name
        self.c_query = c_query
        self.c_kv = c_kv
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.q_head_dim = head_dim
        self.v_head_dim = head_dim
        self.use_rope = True
        self.learnable_rope = True
        assert head_dim % 2 == 0
        assert (num_heads * head_dim) % 8 == 0
        self.norm1 = torch.nn.RMSNorm(c_query, eps=1e-6)
        self.q_proj = torch.nn.Linear(c_query, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(c_kv, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(c_kv, num_heads * head_dim, bias=False)
        self.out_proj = torch.nn.Linear(num_heads * head_dim, c_query, bias=False)
        self.rope_freqs = torch.nn.Parameter(init_rope_freqs(num_heads, head_dim // 2))

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        for name, param in self.named_parameters():
            if "norm" in name or name == "rope_freqs":
                reg_dict["noreg"].append(param)
            else:
                reg_dict["normal_attn"].append(param)

    def forward(self, queries, cand_mask, kv, kv_mask, positions: SparsePositions):
        """queries [N,Q,Cq], cand_mask [N,Q] (unused here, padded queries produce garbage), kv [N,S,Ckv], kv_mask [N,S,1]."""
        batch_size, num_queries, _ = queries.shape
        seq_len = kv.shape[1]
        q = self.q_proj(self.norm1(queries)).view(batch_size, num_queries, self.num_heads, self.q_head_dim)
        k = self.k_proj(kv).view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        v = self.v_proj(kv).view(batch_size, seq_len, self.num_heads, self.v_head_dim)
        cos_q, sin_q = positions.cand_tables(self.rope_freqs)
        cos_k, sin_k = positions.token_tables(self.rope_freqs)
        q, k = apply_learnable_rotary_emb(q, k, cos_q, sin_q, cos_k, sin_k)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=_key_padding_bias(kv_mask, q.dtype), dropout_p=0.0, scale=1.0 / math.sqrt(self.q_head_dim),
        )
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_queries, self.num_heads * self.v_head_dim)
        return self.out_proj(out)


class CandidateDecoder(torch.nn.Module):
    """Candidates read the encoded position through cross-attention, they don't talk to each other.
    candidate = Linear(own features) + Linear(global pooling of the trunk) -> layers of
    [RMSNorm -> cross-attention -> SwiGLU FFN] -> RMSNorm -> Linear -> 6 policy logits + 2 futurepos."""

    def __init__(self, config: modelconfigs.ModelConfig, c_trunk: int, activation: str):
        super().__init__()
        self.c_dec = config["candidate_decoder_channels"]
        self.num_layers = config["candidate_decoder_layers"]
        self.num_heads = config["transformer_heads"]
        self.head_dim = self.c_dec // self.num_heads
        self.ffn_channels = config.get("candidate_decoder_ffn_channels", config["transformer_ffn_channels"])
        self.activation = activation
        self.c_trunk = c_trunk

        self.linear_cand = torch.nn.Linear(NUM_CAND_FEATURES, self.c_dec, bias=True)
        self.linear_ground = torch.nn.Linear(NUM_GROUND_FEATURES, self.c_dec, bias=True)
        self.linear_pool = torch.nn.Linear(3 * c_trunk, self.c_dec, bias=False)

        ffn_config = dict(config)
        ffn_config["transformer_ffn_channels"] = self.ffn_channels
        ffn_config["transformer_ffn_depthwise_conv"] = False
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(CrossAttentionBlock(
                name=f"decoder.attn{i+1}", c_query=self.c_dec, c_kv=c_trunk, num_heads=self.num_heads, head_dim=self.head_dim,
            ))
            self.layers.append(TransformerFFNBlock(
                name=f"decoder.ffn{i+1}", c_main=self.c_dec, config=ffn_config, activation=activation, use_swiglu=True,
            ))
        self.norm_final = torch.nn.RMSNorm(self.c_dec, eps=1e-6)
        self.linear_out = torch.nn.Linear(self.c_dec, NUM_POLICY_OUTPUTS + NUM_FUTUREPOS_OUTPUTS, bias=False)

    def initialize(self):
        init_weights(self.linear_cand.weight, self.activation, scale=0.8)
        torch.nn.init.zeros_(self.linear_cand.bias)
        init_weights(self.linear_ground.weight, self.activation, scale=0.8)
        torch.nn.init.zeros_(self.linear_ground.bias)
        init_weights(self.linear_pool.weight, self.activation, scale=0.6)
        init_weights(self.linear_out.weight, "identity", scale=0.3)

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        reg_dict["output"].append(self.linear_cand.weight)
        reg_dict["output_noreg"].append(self.linear_cand.bias)
        reg_dict["output"].append(self.linear_ground.weight)
        reg_dict["output_noreg"].append(self.linear_ground.bias)
        reg_dict["output"].append(self.linear_pool.weight)
        for layer in self.layers:
            layer.add_reg_dict(reg_dict)
        reg_dict["noreg"].append(self.norm_final.weight)
        reg_dict["output"].append(self.linear_out.weight)

    def forward(self, trunk, mask, num_tokens, board_size, cand_feats, ground_feats, cand_mask, positions):
        """trunk [N,S,C] (after the final norm and activation, zero at padding), cand_feats [N,M,8],
        ground_feats [N,1,1], cand_mask [N,M+1]. Returns [N,M+1,8]."""
        mean = masked_mean_pool(trunk, mask, num_tokens)
        # The activation is > -1, so padding at -1 doesn't affect the max (same trick as KataGPool)
        max_pool = torch.max(trunk + (mask - 1.0), dim=1).values
        pooled = torch.cat([mean, mean * (board_area_offset(board_size) / 10.0), max_pool], dim=1)
        pooled_proj = self.linear_pool(pooled).unsqueeze(1)

        queries = torch.cat([self.linear_cand(cand_feats), self.linear_ground(ground_feats)], dim=1) + pooled_proj
        for i in range(0, len(self.layers), 2):
            queries = queries + self.layers[i](queries, cand_mask, trunk, mask, positions)
            queries = queries + self.layers[i + 1](queries, cand_mask.unsqueeze(-1), None, None, None)
        return self.linear_out(self.norm_final(queries))


class SparseValueHead(torch.nn.Module):
    """KataGo's value head on the token sequence. The board area in the pooling comes from the board size,
    not from the mask. Ownership is one logit per POINT token: the probability that the opponent of the owner
    of the dot captures it by the end of the game."""

    def __init__(self, c_in, c_v1, c_v2, c_sv2, num_scorebeliefs, activation, pos_len_x, pos_len_y, scorebelief_offset=0.0):
        super().__init__()
        self.activation = activation
        self.c_v1 = c_v1
        self.linear1 = torch.nn.Linear(c_in, c_v1, bias=False)
        self.bias1 = torch.nn.Parameter(torch.zeros(c_v1))
        self.act1 = act(activation)
        self.linear2 = torch.nn.Linear(3 * c_v1, c_v2, bias=True)
        self.act2 = act(activation)
        self.linear_valuehead = torch.nn.Linear(c_v2, 3, bias=True)
        self.linear_miscvaluehead = torch.nn.Linear(c_v2, 10, bias=True)
        self.linear_moremiscvaluehead = torch.nn.Linear(c_v2, 8, bias=True)
        self.linear_ownership = torch.nn.Linear(c_v1, 1, bias=False)

        self.scorebelief_mid = pos_len_x * pos_len_y + EXTRA_SCORE_DISTR_RADIUS
        self.scorebelief_len = self.scorebelief_mid * 2
        self.num_scorebeliefs = num_scorebeliefs
        self.c_sv2 = c_sv2
        self.linear_s2 = torch.nn.Linear(3 * c_v1, c_sv2, bias=True)
        self.linear_s2off = torch.nn.Linear(1, c_sv2, bias=False)
        self.linear_s3 = torch.nn.Linear(c_sv2, num_scorebeliefs, bias=True)
        self.linear_smix = torch.nn.Linear(3 * c_v1, num_scorebeliefs, bias=True)
        self.register_buffer("score_belief_offset_vector", torch.tensor(
            [float(i - self.scorebelief_mid) + scorebelief_offset for i in range(self.scorebelief_len)], dtype=torch.float32,
        ), persistent=False)
        self.register_buffer("score_belief_offset_bias_vector", torch.tensor(
            [0.05 * (float(i - self.scorebelief_mid) + scorebelief_offset) for i in range(self.scorebelief_len)], dtype=torch.float32,
        ), persistent=False)

    def initialize(self):
        bias_scale = 0.2
        init_weights(self.linear1.weight, self.activation, scale=1.0)
        init_weights(self.linear2.weight, self.activation, scale=1.0)
        init_weights(self.linear2.bias, self.activation, scale=bias_scale, fan_tensor=self.linear2.weight)
        init_weights(self.linear_valuehead.weight, "identity", scale=1.0)
        init_weights(self.linear_valuehead.bias, "identity", scale=bias_scale, fan_tensor=self.linear_valuehead.weight)
        init_weights(self.linear_miscvaluehead.weight, "identity", scale=1.0)
        init_weights(self.linear_miscvaluehead.bias, "identity", scale=bias_scale, fan_tensor=self.linear_miscvaluehead.weight)
        init_weights(self.linear_moremiscvaluehead.weight, "identity", scale=1.0)
        init_weights(self.linear_moremiscvaluehead.bias, "identity", scale=bias_scale, fan_tensor=self.linear_moremiscvaluehead.weight)
        init_weights(self.linear_ownership.weight, "identity", scale=0.2)
        init_weights(self.linear_s2.weight, self.activation, scale=1.0)
        init_weights(self.linear_s2.bias, self.activation, scale=1.0, fan_tensor=self.linear_s2.weight)
        init_weights(self.linear_s2off.weight, self.activation, scale=1.0, fan_tensor=self.linear_s2.weight)
        scorebelief_output_scale = 0.5
        init_weights(self.linear_s3.weight, "identity", scale=scorebelief_output_scale)
        init_weights(self.linear_s3.bias, "identity", scale=scorebelief_output_scale * bias_scale, fan_tensor=self.linear_s3.weight)
        init_weights(self.linear_smix.weight, "identity", scale=1.0)
        init_weights(self.linear_smix.bias, "identity", scale=bias_scale, fan_tensor=self.linear_smix.weight)

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        reg_dict["output"].append(self.linear1.weight)
        reg_dict["output_noreg"].append(self.bias1)
        for linear in [self.linear2, self.linear_valuehead, self.linear_miscvaluehead, self.linear_moremiscvaluehead,
                       self.linear_s2, self.linear_s3, self.linear_smix]:
            reg_dict["output"].append(linear.weight)
            reg_dict["output_noreg"].append(linear.bias)
        reg_dict["output"].append(self.linear_ownership.weight)
        reg_dict["output"].append(self.linear_s2off.weight)

    def forward(self, trunk, mask, num_tokens, board_size, num_points):
        """trunk [N,S,C], mask [N,S,1]. The POINT tokens are at [1, 1+num_points)."""
        outv1 = self.act1(self.linear1(trunk) + self.bias1.view(1, 1, -1)) * mask
        mean = masked_mean_pool(outv1, mask, num_tokens)
        offset = board_area_offset(board_size)
        pooled = torch.cat([mean, mean * (offset / 10.0), mean * (offset * offset / 100.0 - 0.1)], dim=1)

        outv2 = self.act2(self.linear2(pooled))
        out_value = self.linear_valuehead(outv2)
        out_miscvalue = self.linear_miscvaluehead(outv2)
        out_moremiscvalue = self.linear_moremiscvaluehead(outv2)
        out_ownership = self.linear_ownership(outv1[:, 1:1 + num_points, :]).squeeze(-1)

        batch_size = trunk.shape[0]
        outsv2 = (
            self.linear_s2(pooled).view(batch_size, 1, self.c_sv2)
            + self.linear_s2off(self.score_belief_offset_bias_vector.view(1, self.scorebelief_len, 1))
        )
        outsv2 = self.act2(outsv2)
        outsv3 = self.linear_s3(outsv2)
        outsmix_logweights = torch.nn.functional.log_softmax(self.linear_smix(pooled), dim=1)
        out_scorebelief_logprobs = torch.nn.functional.log_softmax(outsv3, dim=1)
        out_scorebelief_logprobs = torch.logsumexp(out_scorebelief_logprobs + outsmix_logweights.view(-1, 1, self.num_scorebeliefs), dim=2)
        return out_value, out_miscvalue, out_moremiscvalue, out_ownership, out_scorebelief_logprobs


class SparseDotsModel(torch.nn.Module):
    def __init__(self, config: modelconfigs.ModelConfig, pos_len_x: int, pos_len_y: int, games=None, katago_git_rev: str = None, katago_backend: str = None):
        super().__init__()
        assert is_sparse_dots_config(config)
        self.config = config
        self.games = games or [Game.DOTS]
        assert self.games == [Game.DOTS], "Sparse models are Dots only"
        self.katago_git_rev = katago_git_rev
        self.katago_backend = katago_backend
        self.pos_len_x = pos_len_x
        self.pos_len_y = pos_len_y
        # Read by train.py around validation, flex attention is never used here
        self.use_flex_attention = False
        self.norm_kind = config["norm_kind"]
        self.block_kind = config["block_kind"]
        self.c_trunk = config["trunk_num_channels"]
        self.c_mid = config["mid_num_channels"]
        self.c_gpool = config["gpool_num_channels"]
        self.c_v1 = config["v1_num_channels"]
        self.c_v2 = config["v2_size"]
        self.c_sv2 = config["sbv2_num_channels"]
        self.num_scorebeliefs = config["num_scorebeliefs"]
        self.num_total_blocks = len(self.block_kind)
        self.activation = config.get("activation", "relu")
        self.has_intermediate_head = False
        self.metadata_encoder = None
        self.trunk_final_rmsnorm = True
        self.trunk_normless = False
        config_error = modelconfigs.sparse_dots_config_error(config)
        assert config_error is None, f"Not a sparse Dots trunk: {config_error}"

        self.td_score_multiplier = 20.0
        self.scoremean_multiplier = 20.0
        self.scorestdev_multiplier = 20.0
        self.lead_multiplier = 20.0
        self.variance_time_multiplier = 40.0
        self.shortterm_value_error_multiplier = 0.25
        self.shortterm_score_error_multiplier = 150.0

        # Embeddings
        self.linear_point = torch.nn.Linear(NUM_POINT_FEATURES, self.c_trunk, bias=True)
        self.global_token = torch.nn.Parameter(torch.zeros(self.c_trunk))
        self.linear_global = torch.nn.Linear(NUM_GLOBAL_FEATURES, self.c_trunk, bias=False)

        # Trunk
        self.blocks = torch.nn.ModuleList()
        for block_name, block_kind in self.block_kind:
            internal_length = modelconfigs.SPARSE_DOTS_TRUNK_BLOCK_KINDS[block_kind]
            block = NestedBottleneckTransformerBlock(
                name=block_name,
                internal_length=internal_length,
                c_main=self.c_trunk,
                c_mid=self.c_mid,
                config=self.config,
                activation=self.activation,
                pos_len_x=pos_len_x,
                pos_len_y=pos_len_y,
                use_swiglu=True,
                use_rope=True,
            )
            for sub in block.blockstack:
                if isinstance(sub, TransformerAttentionBlock):
                    sub.__class__ = SparseRoPEAttentionBlock
                    sub.rope_freqs = torch.nn.Parameter(init_rope_freqs(*sub.rope_freqs.shape[:2]))
            self.blocks.append(block)
        self.norm_trunkfinal = SparseRMSNormMask(self.c_trunk)
        self.act_trunkfinal = act(self.activation)

        # Heads
        self.decoder = CandidateDecoder(config, self.c_trunk, self.activation)
        self.value_head = SparseValueHead(
            self.c_trunk, self.c_v1, self.c_v2, self.c_sv2, self.num_scorebeliefs, self.activation, pos_len_x, pos_len_y,
            scorebelief_offset=0.0,
        )

    @property
    def device(self):
        return self.linear_global.weight.device

    def initialize(self):
        with torch.no_grad():
            init_weights(self.linear_point.weight, self.activation, scale=0.8)
            torch.nn.init.zeros_(self.linear_point.bias)
            torch.nn.init.normal_(self.global_token, std=0.1)
            init_weights(self.linear_global.weight, self.activation, scale=0.6)

            if self.norm_kind == "fixup":
                fixup_scale = 1.0 / math.sqrt(self.num_total_blocks)
                for block in self.blocks:
                    block.initialize(fixup_scale=fixup_scale)
            elif self.norm_kind in ("fixscale", "fixbrenorm", "fixscaleonenorm"):
                for i, block in enumerate(self.blocks):
                    block.initialize(fixup_scale=1.0 / math.sqrt(i + 1.0))
            else:
                for block in self.blocks:
                    block.initialize(fixup_scale=1.0)

            self.decoder.initialize()
            self.value_head.initialize()

    def get_norm_kind(self):
        return self.norm_kind

    def get_has_intermediate_head(self) -> bool:
        return False

    def get_has_metadata_encoder(self) -> bool:
        return False

    def supports_per_block_compile(self) -> bool:
        return False

    def compile_with_dynamic_shapes(self) -> bool:
        # The numbers of points and candidates change from batch to batch
        return True

    def disable_fused_attention_kernels(self) -> bool:
        return False

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        for key in ["input", "input_noreg", "normal", "normal_attn", "normal_gab", "gab_mlp", "tab_module",
                    "normal_gamma", "noreg", "output", "output_noreg"]:
            reg_dict[key] = []
        reg_dict["input"].append(self.linear_point.weight)
        reg_dict["input_noreg"].append(self.linear_point.bias)
        reg_dict["input_noreg"].append(self.global_token)
        reg_dict["input"].append(self.linear_global.weight)
        for block in self.blocks:
            block.add_reg_dict(reg_dict)
        self.norm_trunkfinal.add_reg_dict(reg_dict)
        self.decoder.add_reg_dict(reg_dict)
        self.value_head.add_reg_dict(reg_dict)

    def set_brenorm_params(self, renorm_avg_momentum: float, rmax: float, dmax: float):
        for block in self.blocks:
            block.set_brenorm_params(renorm_avg_momentum, rmax, dmax)

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        for block in self.blocks:
            block.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)

    def embed(self, batch: Dict[str, torch.Tensor]):
        """Returns tokens [N,S,C], mask [N,S,1], number of tokens [N], positions, board size [N,2] float."""
        board_size = batch["boardSizeN"].to(torch.float32)
        input_global = batch["globalInputNC"].to(torch.float32)
        n = board_size.shape[0]
        num_points = batch["pointCoordsNP"].shape[1]
        point_mask = count_mask(batch["pointCountN"], num_points)

        point_tokens = self.linear_point(point_features(batch["pointCoordsNP"], batch["pointDataNP"], board_size))
        global_tokens = self.global_token.view(1, 1, -1).expand(n, 1, -1)
        tokens = torch.cat([global_tokens, point_tokens], dim=1) + self.linear_global(input_global).unsqueeze(1)
        mask = torch.cat([point_mask.new_ones((n, 1)), point_mask], dim=1).unsqueeze(-1)
        tokens = tokens * mask
        num_tokens = mask.sum(dim=(1, 2))

        positions = SparsePositions(
            point_xy=batch["pointCoordsNP"].to(torch.float32),
            cand_xy=batch["candCoordsNM"].to(torch.float32),
        )
        return tokens, mask, num_tokens, positions, board_size

    def forward(self, batch: Dict[str, torch.Tensor], extra_outputs: Optional[ExtraOutputs] = None):
        """Returns a tuple with one tuple of outputs (like Model.forward):
        policy logits [N,6,M+1] (the last candidate is grounding, masked ones get -5000),
        value [N,3], miscvalue [N,10], moremiscvalue [N,8], ownership logits [N,P],
        futurepos pretanh [N,2,M], scorebelief log probs [N,L]."""
        tokens, mask, num_tokens, positions, board_size = self.embed(batch)
        num_points = batch["pointCoordsNP"].shape[1]
        num_cands = batch["candCoordsNM"].shape[1]
        mask_sum_hw = num_tokens.view(-1, 1, 1)
        mask_sum = torch.sum(mask)
        block_shared_data = {SPARSE_POSITIONS: positions}

        if extra_outputs is not None:
            extra_outputs.report("trunk_after_embedding", tokens)
        out = tokens
        for i, block in enumerate(self.blocks):
            out = out + block(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs, block_shared_data=block_shared_data)
            if extra_outputs is not None:
                extra_outputs.report(f"trunk_after_block{i}", out)
        out = self.act_trunkfinal(self.norm_trunkfinal(out, mask, num_tokens)) * mask
        if extra_outputs is not None:
            extra_outputs.report("trunkfinal", out)

        cand_mask = torch.cat([
            count_mask(batch["candCountN"], num_cands),
            batch["groundN"][:, 0:1].to(torch.float32),
        ], dim=1)
        # The decoder runs under the autocast of the caller like the trunk (it has as many rows as the trunk or more),
        # its logits and the value head are in FP32
        cand_out = self.decoder(
            out, mask, num_tokens, board_size,
            cand_features(batch["candCoordsNM"], batch["candDataNM"], board_size),
            ground_features(batch["groundN"]),
            cand_mask,
            positions,
        ).float()
        with autocast("cuda", enabled=False):
            out = out.float()
            out_policy = cand_out[..., :NUM_POLICY_OUTPUTS].permute(0, 2, 1) - (1.0 - cand_mask.unsqueeze(1)) * 5000.0
            out_futurepos = cand_out[:, :num_cands, NUM_POLICY_OUTPUTS:].permute(0, 2, 1)
            out_value, out_miscvalue, out_moremiscvalue, out_ownership, out_scorebelief = self.value_head(
                out, mask, num_tokens, board_size, num_points,
            )
        return ((out_policy, out_value, out_miscvalue, out_moremiscvalue, out_ownership, out_futurepos, out_scorebelief),)

    def float32ify_output(self, outputs_byheads):
        return tuple(tuple(x.to(torch.float32) for x in outputs) for outputs in outputs_byheads)

    def postprocess_output(self, outputs_byheads):
        return tuple(self.postprocess_single_heads_output(outputs) for outputs in outputs_byheads)

    def postprocess_single_heads_output(self, outputs):
        (out_policy, out_value, out_miscvalue, out_moremiscvalue, out_ownership, out_futurepos, out_scorebelief) = outputs
        td_value_logits = torch.stack((out_miscvalue[:, 4:7], out_miscvalue[:, 7:10], out_moremiscvalue[:, 2:5]), dim=1)
        return (
            out_policy,  # N, 6, M+1
            out_value,  # N, {win,loss,noresult}
            td_value_logits,  # N, {long, mid, short} {win,loss,noresult}
            out_moremiscvalue[:, 5:8] * self.td_score_multiplier,  # N, {long, mid, short}
            out_ownership,  # N, P logits of capture by the opponent of the owner
            out_futurepos,  # N, 2, M pretanh
            out_miscvalue[:, 0] * self.scoremean_multiplier,
            SoftPlusWithGradientFloorFunction.apply(out_miscvalue[:, 1], 0.05, False) * self.scorestdev_multiplier,
            out_miscvalue[:, 2] * self.lead_multiplier,
            SoftPlusWithGradientFloorFunction.apply(out_miscvalue[:, 3], 0.05, False) * self.variance_time_multiplier,
            SoftPlusWithGradientFloorFunction.apply(out_moremiscvalue[:, 0], 0.05, True) * self.shortterm_value_error_multiplier,
            SoftPlusWithGradientFloorFunction.apply(out_moremiscvalue[:, 1], 0.05, True) * self.shortterm_score_error_multiplier,
            out_scorebelief,
        )


def create_model(config: modelconfigs.ModelConfig, pos_len_x: int, pos_len_y: int, games=None, katago_git_rev: str = None, katago_backend: str = None):
    """The sparse model for sparse Dots configs, otherwise the dense one."""
    if is_sparse_dots_config(config):
        return SparseDotsModel(config, pos_len_x, pos_len_y, games, katago_git_rev, katago_backend)
    from ..train.model_pytorch import Model
    return Model(config, pos_len_x, pos_len_y, games, katago_git_rev, katago_backend)
