"""Tests of the sparse Dots model (model version 18), see docs/DotsSparseNN.md and DotsSparseNNPlan.md stage 2."""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from katago.train.metrics_sparse_dots import SparseDotsMetrics
from katago.train.model_pytorch import Game, apply_learnable_rotary_emb
from katago.train.model_sparse_dots import SparsePositions, apply_symmetry_to_batch, count_mask, create_model, trim_batch
from katago.train.sparse_dots_bin import load_sparse_dots_bin

from sparse_dots_testutils import random_batch, tiny_sparse_config

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_model(seed=0, area_rope="mean", decoder_layers=2, x=9, y=7):
    torch.manual_seed(seed)
    model = create_model(tiny_sparse_config(area_rope, decoder_layers), x, y, [Game.DOTS])
    model.initialize()
    # Random non-trivial values for the parameters that are zero-initialized
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "bias" in name or "beta" in name or "gamma" in name or name == "global_token":
                param.add_(0.1 * torch.randn_like(param))
    model.eval()
    return model


def run(model, batch):
    with torch.no_grad():
        return model(batch)[0]


def valid_outputs(batch, outputs, row):
    """Outputs of one row restricted to its valid entries, in a layout independent of the padding."""
    policy, value, misc, moremisc, ownership, futurepos, scorebelief = outputs
    num_cands = int(batch["candCountN"][row])
    num_points = int(batch["pointCountN"][row])
    ground_allowed = batch["groundN"][row, 0] > 0
    result = {
        "policy_cands": policy[row, :, :num_cands],
        "value": value[row], "misc": misc[row], "moremisc": moremisc[row],
        "ownership": ownership[row, :num_points],
        "futurepos": futurepos[row, :, :num_cands],
        "scorebelief": scorebelief[row],
    }
    if ground_allowed:
        result["policy_ground"] = policy[row, :, -1]
    return result


def assert_close(a, b, atol=2e-5):
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].shape == b[key].shape, key
        assert torch.allclose(a[key], b[key], atol=atol, rtol=1e-4), (key, (a[key] - b[key]).abs().max())


def test_forward_shapes_and_finiteness():
    model = make_model()
    batch = random_batch(1, n=6)
    policy, value, misc, moremisc, ownership, futurepos, scorebelief = run(model, batch)
    n = 6
    num_cands = batch["candCoordsNM"].shape[1]
    assert policy.shape == (n, 6, num_cands + 1)
    assert value.shape == (n, 3) and misc.shape == (n, 10) and moremisc.shape == (n, 8)
    assert ownership.shape == (n, batch["pointCoordsNP"].shape[1])
    assert futurepos.shape == (n, 2, num_cands)
    assert scorebelief.shape == (n, model.value_head.scorebelief_len)
    for t in (policy, value, misc, moremisc, ownership, futurepos, scorebelief):
        assert torch.isfinite(t).all()
    # Masked candidates are pushed far below the valid ones
    mask = torch.cat([count_mask(batch["candCountN"], num_cands), batch["groundN"][:, 0:1]], dim=1)
    assert (policy[:, 0][mask == 0] < -1000).all()


@pytest.mark.parametrize("area_rope", ["mean", "none"])
def test_padding_invariance(area_rope):
    model = make_model(area_rope=area_rope)
    batch = random_batch(2, n=5)
    padded = random_batch(2, n=5, pad_extra=4)
    out = run(model, batch)
    out_padded = run(model, padded)
    for row in range(5):
        assert_close(valid_outputs(batch, out, row), valid_outputs(padded, out_padded, row))
    # Each row alone gives the same as in the batch
    for row in range(5):
        single = trim_batch({k: v[row:row + 1] for k, v in batch.items()})
        assert_close(valid_outputs(single, run(model, single), 0), valid_outputs(batch, out, row))


def permute_row(batch, rng):
    """Permutes the points, the areas (with their vertices), the vertices and the candidates of each row."""
    result = {k: v.clone() for k, v in batch.items()}
    perms = []
    for row in range(batch["boardSizeN"].shape[0]):
        num_points = int(batch["pointCountN"][row])
        num_areas = int(batch["areaCountN"][row])
        num_vertices = int(batch["vertexCountN"][row])
        num_cands = int(batch["candCountN"][row])
        pp = torch.from_numpy(rng.permutation(num_points))
        ap = torch.from_numpy(rng.permutation(num_areas))
        vp = torch.from_numpy(rng.permutation(num_vertices))
        cp = torch.from_numpy(rng.permutation(num_cands))
        result["pointCoordsNP"][row, :num_points] = batch["pointCoordsNP"][row, pp]
        result["pointDataNP"][row, :num_points] = batch["pointDataNP"][row, pp]
        result["pointOwnershipTargetsNP"][row, :num_points] = batch["pointOwnershipTargetsNP"][row, pp]
        result["areaDataNA"][row, :num_areas] = batch["areaDataNA"][row, ap]
        inverse_ap = torch.argsort(ap)
        result["vertexCoordsNV"][row, :num_vertices] = batch["vertexCoordsNV"][row, vp]
        result["vertexAreaNV"][row, :num_vertices] = inverse_ap[batch["vertexAreaNV"][row, vp].long()].to(batch["vertexAreaNV"].dtype)
        for key in ["candCoordsNM", "candDataNM", "candPolicyTargetsNM", "candFutureposTargetsNM"]:
            result[key][row, :num_cands] = batch[key][row, cp]
        perms.append((pp, cp))
    return result, perms


def test_token_order_equivariance():
    """The model has no notion of token or slot order: permuting points, areas, vertices and candidates
    permutes the ownership and the candidate outputs the same way and keeps everything else."""
    model = make_model()
    batch = random_batch(3, n=5)
    permuted, perms = permute_row(batch, np.random.default_rng(0))
    out = run(model, batch)
    out_permuted = run(model, permuted)
    for row, (pp, cp) in enumerate(perms):
        expected = valid_outputs(batch, out, row)
        expected["ownership"] = expected["ownership"][pp]
        expected["policy_cands"] = expected["policy_cands"][:, cp]
        expected["futurepos"] = expected["futurepos"][:, cp]
        assert_close(valid_outputs(permuted, out_permuted, row), expected)


def test_symmetry_of_coordinates():
    batch = random_batch(4, n=4, x_size=8, y_size=8)
    for symmetry in range(8):
        transformed = apply_symmetry_to_batch(batch, symmetry)
        for key in ["pointCoordsNP", "vertexCoordsNV", "candCoordsNM"]:
            coords = transformed[key]
            assert (coords >= 0).all() and (coords < 8).all()
        # All symmetries without transposition are involutions
        if symmetry < 4:
            back = apply_symmetry_to_batch(transformed, symmetry)
            for key in ["pointCoordsNP", "vertexCoordsNV", "candCoordsNM"]:
                assert torch.equal(back[key], batch[key])
    # Flip X on a known point
    flipped = apply_symmetry_to_batch(batch, 2)
    assert torch.equal(flipped["candCoordsNM"][..., 0], (8 - 1 - batch["candCoordsNM"][..., 0]).to(torch.int8))
    with pytest.raises(AssertionError):
        apply_symmetry_to_batch(random_batch(4, n=2, x_size=9, y_size=7), 4)


def test_mean_rope_identity():
    """The attention logit of a point query to an area key equals the mean of the logits as if the content of the
    area stood at each vertex of its boundary: rotation is linear in the cos/sin tables."""
    torch.manual_seed(0)
    heads, pairs = 3, 4
    freqs = torch.randn(heads, pairs, 2)
    vertices = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 2.0], [2.0, 1.0], [5.0, 7.0]])
    point = torch.tensor([[4.0, 6.0]])
    positions = SparsePositions(
        point_xy=point.view(1, 1, 2),
        vertex_xy=vertices.view(1, -1, 2),
        vertex_weight=torch.full((1, vertices.shape[0]), 1.0 / vertices.shape[0]),
        vertex_area=torch.zeros((1, vertices.shape[0]), dtype=torch.long),
        num_areas=1,
        cand_xy=torch.zeros((1, 0, 2)),
        area_rope="mean",
    )
    cos, sin = positions.token_tables(freqs)  # [1, 3, H, P]: global, point, area
    assert torch.equal(cos[0, 0], torch.zeros(heads, pairs)) and torch.equal(sin[0, 0], torch.zeros(heads, pairs))
    q = torch.randn(1, 1, heads, 2 * pairs)
    k = torch.randn(1, 1, heads, 2 * pairs)
    q_rot, k_rot = apply_learnable_rotary_emb(q, k, cos[:, 1:2], sin[:, 1:2], cos[:, 2:3], sin[:, 2:3])
    logit_area = (q_rot * k_rot).sum(-1)  # [1, 1, H]

    logits = []
    for v in vertices:
        angles = v[0] * freqs[:, :, 0] + v[1] * freqs[:, :, 1]
        _, k_v = apply_learnable_rotary_emb(q, k, cos[:, 1:2], sin[:, 1:2], torch.cos(angles).view(1, 1, heads, pairs), torch.sin(angles).view(1, 1, heads, pairs))
        logits.append((q_rot * k_v).sum(-1))
    assert torch.allclose(logit_area, torch.stack(logits).mean(0), atol=1e-5)

    # For a point the tables are the plain RoPE of its position
    angles = point[0, 0] * freqs[:, :, 0] + point[0, 1] * freqs[:, :, 1]
    assert torch.allclose(cos[0, 1], torch.cos(angles)) and torch.allclose(sin[0, 1], torch.sin(angles))


def one_hot_targets(batch, seed):
    """Deterministic one-hot policy targets so that the tiny set can be fit exactly."""
    rng = np.random.default_rng(seed)
    batch = {k: v.clone() for k, v in batch.items()}
    batch["candPolicyTargetsNM"].zero_()
    batch["groundPolicyTargetsN"].zero_()
    for row in range(batch["boardSizeN"].shape[0]):
        num_cands = int(batch["candCountN"][row])
        ground = batch["groundN"][row, 0] > 0
        choices = num_cands + (1 if ground else 0)
        choice = int(rng.integers(choices))
        if choice < num_cands:
            batch["candPolicyTargetsNM"][row, choice, 0] = 10
        else:
            batch["groundPolicyTargetsN"][row, 0] = 10
    return batch


def compute_losses(model, metrics, batch):
    return metrics.metrics_dict_batchwise(
        model, model.postprocess_output(model(batch)), None, batch, is_training=True, soft_policy_weight_scale=8.0,
        disable_optimistic_policy=False, meta_kata_only_soft_policy=False, value_loss_scale=0.6,
        td_value_loss_scales=[0.6, 0.6, 0.6], variance_time_loss_scale=1.0, main_loss_scale=1.0,
        intermediate_loss_scale=None, include_model_norms=False,
    )


def test_overfit_tiny_set():
    torch.manual_seed(0)
    model = create_model(tiny_sparse_config(), 9, 7, [Game.DOTS])
    model.initialize()
    model.train()
    batch = one_hot_targets(random_batch(5, n=8, score_len=model.value_head.scorebelief_len), 5)
    metrics = SparseDotsMetrics(1, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    first = None
    for step in range(400):
        optimizer.zero_grad()
        results = compute_losses(model, metrics, batch)
        if first is None:
            first = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in results.items()}
        results["loss_sum"].backward()
        optimizer.step()
    last = {k: float(v) for k, v in compute_losses(model, metrics, batch).items()}
    # The policy of the player is memorized, the ownership and futurepos too
    assert last["p0loss_sum"] < 0.05 * first["p0loss_sum"], (first["p0loss_sum"], last["p0loss_sum"])
    assert last["pacc1_sum"] == pytest.approx(8.0)
    assert last["oloss_sum"] < 0.1 * first["oloss_sum"], (first["oloss_sum"], last["oloss_sum"])
    assert last["fploss_sum"] < 0.2 * first["fploss_sum"], (first["fploss_sum"], last["fploss_sum"])
    assert last["loss_sum"] < 0.3 * first["loss_sum"]


@pytest.mark.parametrize("area_rope,decoder_layers", [("mean", 2), ("none", 1)])
def test_export_round_trip(tmp_path, area_rope, decoder_layers):
    model = make_model(seed=1, area_rope=area_rope, decoder_layers=decoder_layers)
    checkpoint = tmp_path / "model.ckpt"
    torch.save({
        "model": model.state_dict(),
        "config": model.config,
        "pos_len_x": model.pos_len_x,
        "pos_len_y": model.pos_len_y,
        "games": [Game.DOTS.name],
    }, checkpoint)
    export_dir = tmp_path / "export"
    subprocess.run([
        sys.executable, os.path.join(PYTHON_DIR, "export_model_pytorch.py"),
        "-checkpoint", str(checkpoint), "-export-dir", str(export_dir), "-model-name", "sparse-roundtrip",
        "-filename-prefix", "model",
    ], check=True, cwd=PYTHON_DIR, capture_output=True)
    loaded, info = load_sparse_dots_bin(str(export_dir / "model.bin"), model.config)
    assert info["name"] == "sparse-roundtrip" and info["version"] == 18
    loaded.eval()

    batch = random_batch(6, n=5)
    out = run(model, batch)
    out_loaded = run(loaded, batch)
    policy, value, misc, moremisc, ownership, futurepos, _ = out
    policy_l, value_l, misc_l, moremisc_l, ownership_l, _, _ = out_loaded
    # Only what the engine uses is exported
    mask = torch.cat([count_mask(batch["candCountN"], batch["candCoordsNM"].shape[1]), batch["groundN"][:, 0:1]], dim=1) > 0
    for channel in (0, 5):
        assert torch.allclose(policy[:, channel][mask], policy_l[:, channel][mask], atol=1e-5)
    assert torch.allclose(value, value_l, atol=1e-5)
    assert torch.allclose(misc[:, 0:4], misc_l[:, 0:4], atol=1e-5)
    assert torch.allclose(moremisc[:, 0:2], moremisc_l[:, 0:2], atol=1e-5)
    point_mask = count_mask(batch["pointCountN"], batch["pointCoordsNP"].shape[1]) > 0
    assert torch.allclose(ownership[point_mask], ownership_l[point_mask], atol=1e-5)
