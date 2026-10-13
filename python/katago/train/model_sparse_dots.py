# Sparse Dots model (model version 18), see docs/DotsSparseNN.md.
#
# A position is a set of tokens instead of a board grid:
#   GLOBAL (one per position, no position), POINT (placed dots outside the area interiors),
#   AREA (completed enclosures, encoded from the set of their boundary vertices).
# Action candidates (reasonable points and grounding) are queries of a decoder with cross-attention
# to the trunk tokens. The trunk is the usual stack of nested bottleneck transformer blocks running
# on the token sequence ("1 x S board" with a mask).
#
# All the geometry (scaled coordinates, distances to the edges, Fourier features, RoPE) is computed
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
NUM_POINT_FEATURES = 12  # owner, 4 edge distances, grounded, boundary of an area, moves ago one-hot
FOURIER_PERIODS = (4, 8, 16, 32, 64)
NUM_VERTEX_FEATURES = 4 + 4 * len(FOURIER_PERIODS)  # 24
NUM_AREA_SCALAR_FEATURES = 10  # owner, insideMy, insideOpp, boundaryLen, grounded, moves ago one-hot
NUM_CAND_FEATURES = 8  # 4 edge distances, gainMy, gainOpp, inMyHouse, inOppHouse
NUM_GROUND_FEATURES = 1  # ground result
NUM_POLICY_OUTPUTS = 6
NUM_FUTUREPOS_OUTPUTS = 2

COORD_SCALE = 1.0 / 40.0
COUNT_SCALE = 1.0 / 20.0

# Keys of the batch dict (training data arrays, docs/DotsSparseNN.md 7.2)
INPUT_KEYS = [
    "boardSizeN", "globalInputNC",
    "pointCountN", "pointCoordsNP", "pointDataNP",
    "areaCountN", "areaDataNA",
    "vertexCountN", "vertexCoordsNV", "vertexAreaNV",
    "candCountN", "candCoordsNM", "candDataNM",
    "groundN",
]
# Padded dimension (1-based axis) of each per-entity array
PADDED_ARRAYS = {
    "pointCoordsNP": "P", "pointDataNP": "P", "pointOwnershipTargetsNP": "P",
    "areaDataNA": "A",
    "vertexCoordsNV": "V", "vertexAreaNV": "V",
    "candCoordsNM": "M", "candDataNM": "M", "candPolicyTargetsNM": "M", "candFutureposTargetsNM": "M",
}
COUNT_KEY_OF_DIM = {"P": "pointCountN", "A": "areaCountN", "V": "vertexCountN", "M": "candCountN"}

SPARSE_POSITIONS = "sparse_positions"
SPARSE_BOUNDARY = "sparse_boundary"


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


def point_features(point_coords: torch.Tensor, point_data: torch.Tensor, point_boundary: torch.Tensor,
                   board_size: torch.Tensor) -> torch.Tensor:
    """point_coords [N,P,2], point_data [N,P,3] (owner, grounded, movesAgo), point_boundary [N,P] (the point is on
    the boundary of an area, always an area of its owner), board_size [N,2]. Returns [N,P,12]."""
    coords = point_coords.to(torch.float32)
    owner = point_data[..., 0:1].to(torch.float32)
    grounded = point_data[..., 1:2].to(torch.float32)
    return torch.cat([
        owner,
        edge_features(coords, board_size.unsqueeze(1)),
        grounded,
        point_boundary.to(torch.float32).unsqueeze(-1),
        moves_ago_one_hot(point_data[..., 2].long()),
    ], dim=-1)


def vertex_point_index(point_coords: torch.Tensor, point_count: torch.Tensor, vertex_coords: torch.Tensor,
                       vertex_count: torch.Tensor) -> torch.Tensor:
    """Index of the POINT at the coordinates of each boundary vertex [N,V], -1 for padding (and for a vertex
    without a point, which never happens: the boundary of an area is made of dots outside the area interiors)."""
    num_points = point_coords.shape[1]
    point_mask = count_mask(point_count, num_points) > 0
    vertex_mask = count_mask(vertex_count, vertex_coords.shape[1]) > 0
    point_key = point_coords[..., 1].long() * 256 + point_coords[..., 0].long()
    point_key = torch.where(point_mask, point_key, torch.full_like(point_key, 1 << 30))
    vertex_key = vertex_coords[..., 1].long() * 256 + vertex_coords[..., 0].long()
    sorted_key, order = torch.sort(point_key, dim=1)
    pos = torch.searchsorted(sorted_key, vertex_key).clamp(max=num_points - 1)
    found = (torch.gather(sorted_key, 1, pos) == vertex_key) & vertex_mask
    return torch.where(found, torch.gather(order, 1, pos), torch.full_like(pos, -1))


class BoundaryLinks:
    """For each boundary vertex: the flat token index (row * S + token) of its AREA and of its POINT.
    Padding vertices point to the extra row num_rows and are masked."""

    def __init__(self, vertex_area, vertex_point, num_points, num_areas):
        n, v = vertex_point.shape
        seq_len = 1 + num_points + num_areas
        self.num_rows = n * seq_len
        self.area_start = 1 + num_points
        self.num_areas = num_areas
        valid = vertex_point >= 0
        row_start = (torch.arange(n, device=vertex_point.device) * seq_len).view(-1, 1)
        self.valid = valid.reshape(-1)
        self.area_flat = torch.where(valid, row_start + self.area_start + vertex_area,
                                     torch.full_like(vertex_area, self.num_rows)).reshape(-1)
        self.point_flat = torch.where(valid, row_start + 1 + vertex_point, torch.zeros_like(vertex_point)).reshape(-1)


def boundary_attention(q, k, v, links: BoundaryLinks, n_rep: int, scale: float) -> torch.Tensor:
    """Attention of each AREA to the POINTs of its boundary only. q [N,S,H,Dq], k [N,S,Hkv,Dq], v [N,S,Hkv,Dv]
    (after RoPE). Returns [N,A,H*Dv] in float32, zeros for areas without vertices."""
    n, seq_len, num_heads, _ = q.shape
    v_head_dim = v.shape[-1]
    q_area = q.reshape(n * seq_len, num_heads, -1)[links.area_flat.clamp(max=links.num_rows - 1)].to(torch.float32)
    k_point = k.reshape(n * seq_len, k.shape[2], -1)[links.point_flat].to(torch.float32)
    v_point = v.reshape(n * seq_len, v.shape[2], -1)[links.point_flat].to(torch.float32)
    if n_rep > 1:
        k_point = k_point.repeat_interleave(n_rep, dim=1)
        v_point = v_point.repeat_interleave(n_rep, dim=1)
    scores = (q_area * k_point).sum(dim=-1) * scale
    scores = scores.masked_fill(~links.valid.unsqueeze(-1), float("-inf"))
    segment = links.area_flat.unsqueeze(-1).expand(-1, num_heads)
    seg_max = scores.new_full((links.num_rows + 1, num_heads), float("-inf")).scatter_reduce(
        0, segment, scores, reduce="amax", include_self=True)
    seg_max = torch.nan_to_num(seg_max.detach(), neginf=0.0)
    weights = torch.exp(scores - seg_max[links.area_flat])
    denom = scores.new_zeros((links.num_rows + 1, num_heads)).index_add(0, links.area_flat, weights)
    out = scores.new_zeros((links.num_rows + 1, num_heads, v_head_dim)).index_add(
        0, links.area_flat, weights.unsqueeze(-1) * v_point)
    out = out / denom.clamp(min=1e-30).unsqueeze(-1)
    out = out[:links.num_rows].view(n, seq_len, num_heads * v_head_dim)
    return out[:, links.area_start:links.area_start + links.num_areas]


def vertex_features(vertex_coords: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
    """vertex_coords [N,V,2], board_size [N,2]. Returns [N,V,24]."""
    coords = vertex_coords.to(torch.float32)
    features = [edge_features(coords, board_size.unsqueeze(1))]
    x = coords[..., 0]
    y = coords[..., 1]
    for period in FOURIER_PERIODS:
        k = 2.0 * math.pi / period
        features.append(torch.stack([torch.sin(k * x), torch.cos(k * x), torch.sin(k * y), torch.cos(k * y)], dim=-1))
    return torch.cat(features, dim=-1)


def area_scalar_features(area_data: torch.Tensor) -> torch.Tensor:
    """area_data [N,A,6] (owner, grounded, movesAgo, insideMy, insideOpp, boundaryLen). Returns [N,A,10]."""
    data = area_data.to(torch.float32)
    return torch.cat([
        data[..., 0:1],
        data[..., 3:4] * COUNT_SCALE,
        data[..., 4:5] * COUNT_SCALE,
        data[..., 5:6] * COUNT_SCALE,
        data[..., 1:2],
        moves_ago_one_hot(area_data[..., 2].long()),
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
    for key in ["pointCoordsNP", "vertexCoordsNV", "candCoordsNM"]:
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
    """Weighted position sets of the tokens and of the candidates, shared by all attention layers.
    Token order: [GLOBAL][POINT x P][AREA x A]. A point has one position, an area has its boundary vertices
    with weights 1/|B| (area_rope == "mean") or no positions at all ("none"). The GLOBAL token and the grounding
    candidate have no positions: their cos/sin tables are zero, so their rotated q/k vanish (NoPE)."""

    def __init__(self, point_xy, vertex_xy, vertex_weight, vertex_area, num_areas, cand_xy, area_rope):
        self.point_xy = point_xy  # [N,P,2] float
        self.vertex_xy = vertex_xy  # [N,V,2] float
        self.vertex_weight = vertex_weight  # [N,V] float, zero for padding
        self.vertex_area = vertex_area  # [N,V] long, clamped to valid indices
        self.num_areas = num_areas
        self.cand_xy = cand_xy  # [N,M,2] float
        self.area_rope = area_rope

    @staticmethod
    def _angles(xy, freqs):
        # xy [..., 2], freqs [H, Pp, 2] -> [..., H, Pp]
        return xy[..., 0, None, None] * freqs[:, :, 0] + xy[..., 1, None, None] * freqs[:, :, 1]

    def token_tables(self, freqs: torch.Tensor):
        """Returns cos, sin of shape [N, S, H, Pp] for the trunk tokens."""
        freqs = freqs.to(torch.float32)
        n = self.point_xy.shape[0]
        num_heads, num_pairs = freqs.shape[0], freqs.shape[1]
        point_angles = self._angles(self.point_xy, freqs)
        point_cos = torch.cos(point_angles)
        point_sin = torch.sin(point_angles)
        area_cos = point_cos.new_zeros((n, self.num_areas, num_heads, num_pairs))
        area_sin = point_cos.new_zeros((n, self.num_areas, num_heads, num_pairs))
        if self.area_rope == "mean" and self.vertex_xy.shape[1] > 0:
            vertex_angles = self._angles(self.vertex_xy, freqs)
            weight = self.vertex_weight[..., None, None]
            index = self.vertex_area[..., None, None].expand(-1, -1, num_heads, num_pairs)
            area_cos = area_cos.scatter_add(1, index, torch.cos(vertex_angles) * weight)
            area_sin = area_sin.scatter_add(1, index, torch.sin(vertex_angles) * weight)
        global_zeros = point_cos.new_zeros((n, 1, num_heads, num_pairs))
        cos = torch.cat([global_zeros, point_cos, area_cos], dim=1)
        sin = torch.cat([global_zeros, point_sin, area_sin], dim=1)
        return cos, sin

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
    Instances are created as TransformerAttentionBlock and switched to this class.
    With area boundary attention the AREA tokens also attend to the POINTs of their own boundary only, with the
    same q/k/v and a separate output projection boundary_out_proj (exported after the trunk)."""

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
        q_rot, k_rot, v_rot = q, k, v

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

        # Areas also attend to the points of their own boundary only, with the same q/k/v
        boundary_out_proj = getattr(self, "boundary_out_proj", None)
        links: Optional[BoundaryLinks] = block_shared_data.get(SPARSE_BOUNDARY)
        if boundary_out_proj is not None and links is not None and links.num_areas > 0:
            area_output = boundary_attention(q_rot, k_rot, v_rot, links, self.n_rep, 1.0 / math.sqrt(self.q_head_dim))
            area_result = boundary_out_proj(area_output.to(result.dtype))
            result = torch.cat([result[:, :links.area_start], result[:, links.area_start:] + area_result], dim=1)
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
        # Same initialization as the learnable RoPE of TransformerAttentionBlock
        num_pairs = head_dim // 2
        log_lo = math.log(1.0 / 50.0)
        log_hi = math.log(1.0)
        init_freqs = (
            torch.exp(torch.empty(num_heads, num_pairs, 2).uniform_(log_lo, log_hi))
            * (torch.randint(0, 2, (num_heads, num_pairs, 2)) * 2 - 1).float()
        )
        self.rope_freqs = torch.nn.Parameter(init_freqs)

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


_SPARSE_TRUNK_BLOCK_KINDS = {
    "bottlenest2transformerrope": (2, False),
    "bottlenest2transformerropesg": (2, True),
    "bottlenest3transformerropesg": (3, True),
}


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
        self.c_vertex = config.get("vertex_encoder_channels", 192)
        self.area_rope = config.get("area_rope", "mean")
        assert self.area_rope in ("mean", "none")
        self.area_boundary_attention = config.get("area_boundary_attention", True)
        self.activation = config.get("activation", "relu")
        self.has_intermediate_head = False
        self.metadata_encoder = None
        self.trunk_final_rmsnorm = True
        self.trunk_normless = False
        assert config.get("trunk_final_rmsnorm", False) and config.get("trunk_rmsnorm_spatial", False), \
            "Sparse Dots models need the spatial RMSNorm trunk tip (-rsnh)"
        assert config.get("rmsnorm_spatial_cgroup_size") is None
        assert config.get("learnable_rope", False), "Sparse Dots models need learnable RoPE"

        self.td_score_multiplier = 20.0
        self.scoremean_multiplier = 20.0
        self.scorestdev_multiplier = 20.0
        self.lead_multiplier = 20.0
        self.variance_time_multiplier = 40.0
        self.shortterm_value_error_multiplier = 0.25
        self.shortterm_score_error_multiplier = 150.0

        # Embeddings
        self.linear_point = torch.nn.Linear(NUM_POINT_FEATURES, self.c_trunk, bias=True)
        self.linear_vertex1 = torch.nn.Linear(NUM_VERTEX_FEATURES, self.c_vertex, bias=True)
        self.act_vertex = act(self.activation)
        self.linear_vertex2 = torch.nn.Linear(self.c_vertex, self.c_vertex, bias=True)
        self.linear_area1 = torch.nn.Linear(2 * self.c_vertex + 1 + NUM_AREA_SCALAR_FEATURES, self.c_trunk, bias=True)
        self.act_area = act(self.activation)
        self.linear_area2 = torch.nn.Linear(self.c_trunk, self.c_trunk, bias=True)
        self.global_token = torch.nn.Parameter(torch.zeros(self.c_trunk))
        self.linear_global = torch.nn.Linear(NUM_GLOBAL_FEATURES, self.c_trunk, bias=False)

        # Trunk
        self.blocks = torch.nn.ModuleList()
        for block_name, block_kind in self.block_kind:
            assert block_kind in _SPARSE_TRUNK_BLOCK_KINDS, f"Block kind {block_kind} is not supported by sparse Dots models"
            internal_length, use_swiglu = _SPARSE_TRUNK_BLOCK_KINDS[block_kind]
            block = NestedBottleneckTransformerBlock(
                name=block_name,
                internal_length=internal_length,
                c_main=self.c_trunk,
                c_mid=self.c_mid,
                config=self.config,
                activation=self.activation,
                pos_len_x=pos_len_x,
                pos_len_y=pos_len_y,
                use_swiglu=use_swiglu,
                use_rope=True,
            )
            for sub in block.blockstack:
                if isinstance(sub, TransformerAttentionBlock):
                    sub.__class__ = SparseRoPEAttentionBlock
                    if self.area_boundary_attention:
                        sub.boundary_out_proj = torch.nn.Linear(
                            sub.num_heads * sub.v_head_dim, sub.out_proj.out_features, bias=False)
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
            init_weights(self.linear_vertex1.weight, self.activation, scale=1.0)
            torch.nn.init.zeros_(self.linear_vertex1.bias)
            init_weights(self.linear_vertex2.weight, self.activation, scale=1.0)
            torch.nn.init.zeros_(self.linear_vertex2.bias)
            init_weights(self.linear_area1.weight, self.activation, scale=1.0)
            torch.nn.init.zeros_(self.linear_area1.bias)
            init_weights(self.linear_area2.weight, self.activation, scale=0.8)
            torch.nn.init.zeros_(self.linear_area2.bias)
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

    def disable_fused_attention_kernels(self) -> bool:
        return False

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        for key in ["input", "input_noreg", "normal", "normal_attn", "normal_gab", "gab_mlp", "tab_module",
                    "normal_gamma", "noreg", "output", "output_noreg"]:
            reg_dict[key] = []
        for linear in [self.linear_point, self.linear_vertex1, self.linear_vertex2, self.linear_area1, self.linear_area2]:
            reg_dict["input"].append(linear.weight)
            reg_dict["input_noreg"].append(linear.bias)
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
        """Returns tokens [N,S,C], mask [N,S,1], number of tokens [N], positions, boundary links (or None),
        board size [N,2] float."""
        board_size = batch["boardSizeN"].to(torch.float32)
        input_global = batch["globalInputNC"].to(torch.float32)
        n = board_size.shape[0]
        num_points = batch["pointCoordsNP"].shape[1]
        num_areas = batch["areaDataNA"].shape[1]
        num_vertices = batch["vertexCoordsNV"].shape[1]
        point_mask = count_mask(batch["pointCountN"], num_points)
        area_mask = count_mask(batch["areaCountN"], num_areas)
        vertex_mask = count_mask(batch["vertexCountN"], num_vertices)

        vertex_point = vertex_point_index(
            batch["pointCoordsNP"], batch["pointCountN"], batch["vertexCoordsNV"], batch["vertexCountN"])
        point_boundary = point_mask.new_zeros(point_mask.shape).scatter_add(
            1, vertex_point.clamp(min=0), (vertex_point >= 0).to(point_mask.dtype)).clamp(max=1.0)
        point_tokens = self.linear_point(
            point_features(batch["pointCoordsNP"], batch["pointDataNP"], point_boundary, board_size))

        # Area encoder: vertex MLP -> mean | max | log(1+|B|) | area scalars -> area MLP
        area_data = batch["areaDataNA"]
        boundary_len = area_data[..., 5].to(torch.float32).clamp(min=1.0)
        vertex_area = batch["vertexAreaNV"].long().clamp(min=0, max=max(num_areas - 1, 0))
        h = self.linear_vertex2(self.act_vertex(self.linear_vertex1(vertex_features(batch["vertexCoordsNV"], board_size))))
        c_vertex = h.shape[-1]
        index = vertex_area.unsqueeze(-1).expand(-1, -1, c_vertex)
        vertex_mask_c = vertex_mask.unsqueeze(-1)
        h_sum = h.new_zeros((n, num_areas, c_vertex)).scatter_add(1, index, h * vertex_mask_c)
        h_mean = h_sum / boundary_len.unsqueeze(-1)
        h_max = h.new_zeros((n, num_areas, c_vertex)).scatter_reduce(
            1, index, h.masked_fill(vertex_mask_c == 0, float("-inf")), reduce="amax", include_self=False,
        )
        # Areas without vertices are padding, keep them finite
        h_max = torch.where(area_mask.unsqueeze(-1) > 0, h_max, torch.zeros_like(h_max))
        area_agg = torch.cat([h_mean, h_max, torch.log1p(boundary_len).unsqueeze(-1), area_scalar_features(area_data)], dim=-1)
        area_tokens = self.linear_area2(self.act_area(self.linear_area1(area_agg)))

        global_tokens = self.global_token.view(1, 1, -1).expand(n, 1, -1)
        tokens = torch.cat([global_tokens, point_tokens, area_tokens], dim=1) + self.linear_global(input_global).unsqueeze(1)
        mask = torch.cat([point_mask.new_ones((n, 1)), point_mask, area_mask], dim=1).unsqueeze(-1)
        tokens = tokens * mask
        num_tokens = mask.sum(dim=(1, 2))

        vertex_weight = vertex_mask / torch.gather(boundary_len, 1, vertex_area)
        positions = SparsePositions(
            point_xy=batch["pointCoordsNP"].to(torch.float32),
            vertex_xy=batch["vertexCoordsNV"].to(torch.float32),
            vertex_weight=vertex_weight,
            vertex_area=vertex_area,
            num_areas=num_areas,
            cand_xy=batch["candCoordsNM"].to(torch.float32),
            area_rope=self.area_rope,
        )
        links = BoundaryLinks(vertex_area, vertex_point, num_points, num_areas) if self.area_boundary_attention else None
        return tokens, mask, num_tokens, positions, links, board_size

    def forward(self, batch: Dict[str, torch.Tensor], extra_outputs: Optional[ExtraOutputs] = None):
        """Returns a tuple with one tuple of outputs (like Model.forward):
        policy logits [N,6,M+1] (the last candidate is grounding, masked ones get -5000),
        value [N,3], miscvalue [N,10], moremiscvalue [N,8], ownership logits [N,P],
        futurepos pretanh [N,2,M], scorebelief log probs [N,L]."""
        tokens, mask, num_tokens, positions, links, board_size = self.embed(batch)
        num_points = batch["pointCoordsNP"].shape[1]
        num_cands = batch["candCoordsNM"].shape[1]
        mask_sum_hw = num_tokens.view(-1, 1, 1)
        mask_sum = torch.sum(mask)
        block_shared_data = {SPARSE_POSITIONS: positions, SPARSE_BOUNDARY: links}

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

        with autocast("cuda", enabled=False):
            out = out.float()
            cand_mask = torch.cat([
                count_mask(batch["candCountN"], num_cands),
                batch["groundN"][:, 0:1].to(torch.float32),
            ], dim=1)
            cand_out = self.decoder(
                out, mask, num_tokens, board_size,
                cand_features(batch["candCoordsNM"], batch["candDataNM"], board_size),
                ground_features(batch["groundN"]),
                cand_mask,
                positions,
            )
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
