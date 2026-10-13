"""The C++ CPU reference of the sparse Dots model (cpp/neuralnet/sparsedotsnn.cpp) matches PyTorch on positions of
random games. Needs a built katago: set KATAGO_SPARSE_TEST_EXE or build cpp/builds/eigen (skipped otherwise)."""

import json
import os
import subprocess
import sys

import pytest
import torch

from katago.train.model_pytorch import Game
from katago.train.model_sparse_dots import create_model
from katago.train.sparse_dots_bin import load_sparse_dots_bin

from sparse_dots_testutils import collate, tiny_sparse_config

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(PYTHON_DIR)


def find_katago():
    candidates = [os.environ.get("KATAGO_SPARSE_TEST_EXE", "")]
    for build in ["eigen", "cuda-sparse"]:
        for name in ["katago.exe", "katago"]:
            candidates.append(os.path.join(REPO_DIR, "cpp", "builds", build, name))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def json_to_row(entry):
    """A position written by `katago dumpsparsenn` as a row for sparse_dots_testutils.collate (dummy targets)."""
    inp = entry["input"]
    cands = [(c["x"], c["y"], c["gainMy"], c["gainOpp"], c["inMyHouse"], c["inOppHouse"]) for c in inp["candidates"]]
    return {
        "boardSizeN": [inp["xSize"], inp["ySize"]],
        "globalInputNC": inp["global"],
        "points": [(p["x"], p["y"], p["owner"], p["grounded"], p["movesAgo"], p["areaCount"], p["areaMovesAgo"],
                    p["insideMy"], p["insideOpp"]) for p in inp["points"]],
        "cands": cands,
        "groundN": [1.0 if inp["groundAllowed"] else 0.0, inp["groundResult"]],
        "candPolicyTargetsNM": [[0, 0] for _ in cands],
        "groundPolicyTargetsN": [0, 0],
        "candFutureposTargetsNM": [[0, 0] for _ in cands],
        "pointOwnershipTargetsNP": [0 for _ in inp["points"]],
        "globalTargetsNC": [0.0] * 64,
    }


def check_parity(model, entries, atol):
    rows = [json_to_row(e) for e in entries]
    batch = collate(rows, pad_extra=2)
    with torch.no_grad():
        policy, value, misc, moremisc, ownership, _, _ = model(batch)[0]
    max_diff = 0.0
    for i, entry in enumerate(entries):
        num_cands = len(entry["input"]["candidates"])
        ground = entry["input"]["groundAllowed"]
        expected_policy = torch.tensor(entry["policy"]).view(-1, 2)
        assert expected_policy.shape[0] == num_cands + (1 if ground else 0)
        got = [policy[i, [0, 5], :num_cands].T]
        if ground:
            got.append(policy[i, [0, 5], -1].view(1, 2))
        got = torch.cat(got, dim=0)
        expected_scores = torch.tensor(entry["scoreValue"])
        got_scores = torch.cat([misc[i, 0:4], moremisc[i, 0:2]])
        num_points = len(entry["input"]["points"])
        pairs = [
            (got, expected_policy),
            (value[i], torch.tensor(entry["value"])),
            (got_scores, expected_scores),
            (ownership[i, :num_points], torch.tensor(entry["ownership"]).view(-1)),
        ]
        for a, b in pairs:
            if a.numel() == 0:
                continue
            diff = (a - b).abs().max().item()
            max_diff = max(max_diff, diff)
            assert torch.allclose(a, b, atol=atol, rtol=1e-4), (i, diff)
    return max_diff


@pytest.mark.parametrize("decoder_layers,x_size,y_size", [(2, 16, 13), (1, 12, 12)])
def test_cpu_reference_matches_pytorch(tmp_path, decoder_layers, x_size, y_size):
    katago = find_katago()
    if katago is None:
        pytest.skip("No built katago, set KATAGO_SPARSE_TEST_EXE")
    torch.manual_seed(3)
    model = create_model(tiny_sparse_config(decoder_layers), x_size, y_size, [Game.DOTS])
    model.initialize()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "bias" in name or "beta" in name or "gamma" in name or name == "global_token":
                param.add_(0.1 * torch.randn_like(param))
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"model": model.state_dict(), "config": model.config, "pos_len_x": x_size, "pos_len_y": y_size,
                "games": [Game.DOTS.name]}, checkpoint)
    export_dir = tmp_path / "export"
    subprocess.run([sys.executable, os.path.join(PYTHON_DIR, "export_model_pytorch.py"), "-checkpoint", str(checkpoint),
                    "-export-dir", str(export_dir), "-model-name", "parity", "-filename-prefix", "model"],
                   check=True, cwd=PYTHON_DIR, capture_output=True)
    output = tmp_path / "positions.json"
    subprocess.run([katago, "dumpsparsenn", "-model", str(export_dir / "model.bin"), "-output", str(output),
                    "-num-positions", "60", "-x-size", str(x_size), "-y-size", str(y_size), "-seed", str(decoder_layers)],
                   check=True, capture_output=True)
    with open(output) as f:
        entries = json.load(f)
    assert len(entries) == 60
    # Positions with areas: some points carry the facts of the areas of their boundaries
    assert any(p["areaCount"] > 0 for e in entries for p in e["input"]["points"])
    assert any(e["symmetry"] != 0 for e in entries)
    loaded, _ = load_sparse_dots_bin(str(export_dir / "model.bin"), model.config)
    loaded.eval()
    check_parity(loaded, entries, atol=1e-4)
