"""Synthetic batches of the sparse Dots format for tests (not real positions, but structurally valid)."""

import random
from typing import Dict, List

import numpy as np
import torch

from katago.train import modelconfigs


def tiny_sparse_config(decoder_layers=1):
    config = dict(modelconfigs.config_of_name["b5c384h6nbttflrs-fson-silu-rsnh"])
    config.update({
        "trunk_num_channels": 64,
        "mid_num_channels": 32,
        "transformer_ffn_channels": 64,
        "transformer_heads": 2,
        "transformer_kv_heads": 2,
        "block_kind": [["block1", "bottlenest2transformerropesg"], ["block2", "bottlenest2transformerropesg"]],
        "v1_num_channels": 16,
        "v2_size": 24,
        "sbv2_num_channels": 16,
        "num_scorebeliefs": 2,
    })
    return modelconfigs.make_sparse_dots_config(config, {
        "candidate_decoder_layers": decoder_layers,
        "candidate_decoder_channels": 32,
        "candidate_decoder_ffn_channels": 64,
    })


def random_row(rng: random.Random, x_size: int, y_size: int) -> Dict:
    cells = [(x, y) for y in range(y_size) for x in range(x_size)]
    rng.shuffle(cells)
    num_points = rng.randint(0, min(14, len(cells) // 3))
    points = cells[:num_points]
    rest = cells[num_points:]
    num_cands = rng.randint(0, min(12, len(rest)))
    cands = rest[:num_cands]
    ground_allowed = 1 if (num_cands == 0 or rng.random() < 0.5) else 0
    row = {
        "boardSizeN": [x_size, y_size],
        "globalInputNC": [rng.uniform(-1, 1) for _ in range(8)],
        # x, y, owner, grounded, movesAgo, areaCount, areaMovesAgo, insideMy, insideOpp
        "points": [(x, y, rng.randint(0, 1), rng.randint(0, 1), rng.randint(0, 5), area_count,
                    rng.randint(0, 5) if area_count else 0, rng.randint(0, 9) * area_count, rng.randint(0, 9) * area_count)
                   for (x, y) in points for area_count in [rng.choice([0, 0, 1, 1, 2])]],
        "cands": [(x, y, rng.randint(-3, 3), rng.randint(-3, 3), rng.randint(0, 1), rng.randint(0, 1)) for (x, y) in cands],
        "groundN": [ground_allowed, rng.uniform(-10, 10)],
    }
    # Targets
    policy = [[rng.randint(0, 50), rng.randint(0, 50)] for _ in cands]
    row["candPolicyTargetsNM"] = policy
    row["groundPolicyTargetsN"] = [rng.randint(0, 50) * ground_allowed, rng.randint(0, 50) * ground_allowed]
    if sum(p[0] for p in policy) + row["groundPolicyTargetsN"][0] == 0:
        if num_cands > 0:
            policy[0][0] = 1
        else:
            row["groundPolicyTargetsN"][0] = 1
    row["candFutureposTargetsNM"] = [[rng.randint(-1, 1), rng.randint(-1, 1)] for _ in cands]
    row["pointOwnershipTargetsNP"] = [rng.randint(0, 1) for _ in points]
    global_targets = [0.0] * 64
    win = rng.random()
    global_targets[0:3] = [win, 1 - win, 0.0]
    global_targets[3] = rng.uniform(-10, 10)
    for i in (4, 8, 12):
        global_targets[i:i + 3] = [win, 1 - win, 0.0]
    global_targets[21] = global_targets[3]
    global_targets[22] = rng.uniform(0, 30)
    global_targets[25] = 1.0
    for i in (26, 27, 28, 29, 33):
        global_targets[i] = 1.0
    row["globalTargetsNC"] = global_targets
    # The two bins of the final score distribution as trainingwrite.cpp makes them: scores floor and floor + 1
    score = rng.uniform(-4, 4)
    floor_score = int(np.floor(score))
    upper = min(99, int(round((score - floor_score) * 100)))
    row["scoreDistr"] = ([floor_score, floor_score + 1], [100 - upper, upper])
    return row


def collate(rows: List[Dict], pad_extra: int = 0) -> Dict[str, torch.Tensor]:
    """A batch of the rows as the loader gives it: the points and candidates padded to the max counts plus
    pad_extra entries of each kind."""
    n = len(rows)
    P = max([len(r["points"]) for r in rows] + [1]) + pad_extra
    M = max([len(r["cands"]) for r in rows] + [1]) + pad_extra
    b = {
        "boardSizeN": np.zeros((n, 2), np.int8),
        "globalInputNC": np.zeros((n, 8), np.float32),
        "pointCountN": np.zeros(n, np.int16),
        "pointCoordsNP": np.zeros((n, P, 2), np.int8),
        "pointDataNP": np.zeros((n, P, 7), np.int16),
        "candCountN": np.zeros(n, np.int16),
        "candCoordsNM": np.zeros((n, M, 2), np.int8),
        "candDataNM": np.zeros((n, M, 4), np.int16),
        "groundN": np.zeros((n, 2), np.float32),
        "candPolicyTargetsNM": np.zeros((n, M, 2), np.int16),
        "groundPolicyTargetsN": np.zeros((n, 2), np.int16),
        "candFutureposTargetsNM": np.zeros((n, M, 2), np.int8),
        "pointOwnershipTargetsNP": np.zeros((n, P), np.int8),
        "globalTargetsNC": np.zeros((n, 64), np.float32),
        "scoreDistrScoresN": np.zeros((n, 2), np.int16),
        "scoreDistrWeightsN": np.zeros((n, 2), np.int8),
    }
    for i, r in enumerate(rows):
        b["boardSizeN"][i] = r["boardSizeN"]
        b["globalInputNC"][i] = r["globalInputNC"]
        b["pointCountN"][i] = len(r["points"])
        for j, point in enumerate(r["points"]):
            b["pointCoordsNP"][i, j] = point[0:2]
            b["pointDataNP"][i, j] = point[2:9]
        b["candCountN"][i] = len(r["cands"])
        for j, c in enumerate(r["cands"]):
            b["candCoordsNM"][i, j] = c[0:2]
            b["candDataNM"][i, j] = c[2:6]
            b["candPolicyTargetsNM"][i, j] = r["candPolicyTargetsNM"][j]
            b["candFutureposTargetsNM"][i, j] = r["candFutureposTargetsNM"][j]
        b["groundN"][i] = r["groundN"]
        b["groundPolicyTargetsN"][i] = r["groundPolicyTargetsN"]
        for j, t in enumerate(r["pointOwnershipTargetsNP"]):
            b["pointOwnershipTargetsNP"][i, j] = t
        b["globalTargetsNC"][i] = r["globalTargetsNC"]
        b["scoreDistrScoresN"][i], b["scoreDistrWeightsN"][i] = r["scoreDistr"]
    return {k: torch.from_numpy(v) for k, v in b.items()}


def random_batch(seed: int, n: int = 4, x_size: int = 9, y_size: int = 7, pad_extra: int = 0):
    rng = random.Random(seed)
    return collate([random_row(rng, x_size, y_size) for _ in range(n)], pad_extra=pad_extra)
