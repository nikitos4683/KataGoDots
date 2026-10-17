"""Files of the sparse Dots training data: the unpadded format, shuffling and batching."""

import os
import random

import numpy as np
import pytest
import torch

import shuffle
from katago.train import sparse_dots_npz
from katago.train.data_processing_sparse_dots import group_by_length, read_npz_training_data_sparse
from sparse_dots_testutils import collate, random_row

SCORE_LEN = 10


def batch_arrays(rows):
    """The batch of the rows as numpy arrays (the points and candidates padded to the maxima)."""
    return {name: tensor.numpy() for name, tensor in collate(rows).items()}


def loaded_batch_arrays(rows):
    """batch_arrays() as the loader gives it from a file: globalTargetsNC padded to the current channel count."""
    batch = batch_arrays(rows)
    batch["globalTargetsNC"] = sparse_dots_npz.pad_global_targets_nc(batch["globalTargetsNC"])
    return batch


def file_arrays(rows):
    """The rows in the current file format, built directly from the rows."""
    def stack(values, dtype, width):
        return np.array(values, dtype=dtype).reshape(-1, width) if values else np.zeros((0, width), dtype)

    arrays = {
        "boardSizeN": np.array([r["boardSizeN"] for r in rows], np.int8),
        "globalInputNC": np.array([r["globalInputNC"] for r in rows], np.float32),
        "pointCountN": np.array([len(r["points"]) for r in rows], np.int16),
        "candCountN": np.array([len(r["cands"]) for r in rows], np.int16),
        "groundN": np.array([r["groundN"] for r in rows], np.float32),
        "groundPolicyTargetsN": np.array([r["groundPolicyTargetsN"] for r in rows], np.int16),
        "globalTargetsNC": np.array([r["globalTargetsNC"] for r in rows], np.float32),
        "scoreDistrScoresN": np.array([r["scoreDistr"][0] for r in rows], np.int16),
        "scoreDistrWeightsN": np.array([r["scoreDistr"][1] for r in rows], np.int8),
        "pointCoordsP": stack([p[0:2] for r in rows for p in r["points"]], np.int8, 2),
        "pointStateP": stack([p[2:7] for r in rows for p in r["points"]], np.int8, 5),
        "pointInsideP": stack([p[7:9] for r in rows for p in r["points"]], np.int16, 2),
        "pointChainP": np.array([p[9] for r in rows for p in r["points"]], np.int16),
        "pointOwnershipTargetsP": np.array([t for r in rows for t in r["pointOwnershipTargetsNP"]], np.int8),
        "candCoordsM": stack([c[0:2] for r in rows for c in r["cands"]], np.int8, 2),
        "candGainM": stack([c[2:4] for r in rows for c in r["cands"]], np.int16, 2),
        "candHouseM": stack([c[4:6] for r in rows for c in r["cands"]], np.int8, 2),
    }
    for prefix, key, dtype, letter in [("policyTarget", "candPolicyTargetsNM", np.int16, "T"),
                                       ("futureposTarget", "candFutureposTargetsNM", np.int8, "F")]:
        nonzero = [[(j, t) for j, t in enumerate(r[key]) if any(t)] for r in rows]
        arrays[prefix + "CountN"] = np.array([len(e) for e in nonzero], np.int16)
        arrays[prefix + "Cand" + letter] = np.array([j for e in nonzero for j, _ in e], np.int16)
        arrays[prefix + "s" + letter] = stack([t for e in nonzero for _, t in e], dtype, 2)
    arrays["globalTargetsNC"] = sparse_dots_npz.pad_global_targets_nc(arrays["globalTargetsNC"])
    return arrays


def make_rows(seed, n):
    rng = random.Random(seed)
    # Distinct first global input to identify the rows
    rows = [random_row(rng, 9, 7) for _ in range(n)]
    for i, row in enumerate(rows):
        row["globalInputNC"][0] = float(i)
    return rows


def assert_arrays_equal(a, b, names):
    for name in names:
        assert a[name].dtype == b[name].dtype, (name, a[name].dtype, b[name].dtype)
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_current_format_has_no_zero_runs():
    arrays = file_arrays(make_rows(4, 30))
    assert set(arrays) == set(sparse_dots_npz.ARRAY_NAMES)
    assert (arrays["policyTargetsT"] != 0).any(axis=1).all()
    assert (arrays["futureposTargetsF"] != 0).any(axis=1).all()


def test_load_current_and_reject_older_files(tmp_path):
    rows = make_rows(1, 13)
    expected = file_arrays(rows)

    current = str(tmp_path / "current.npz")
    sparse_dots_npz.save_sparse_npz(current, file_arrays(rows))
    assert_arrays_equal(sparse_dots_npz.load_sparse_npz(current), expected, sparse_dots_npz.ARRAY_NAMES)

    older = str(tmp_path / "older.npz")
    arrays = file_arrays(rows)
    del arrays["pointChainP"]
    np.savez_compressed(older, **arrays)
    with pytest.raises(ValueError, match="older format"):
        sparse_dots_npz.load_sparse_npz(older)


def test_select_rows_and_collate():
    rows = make_rows(2, 11)
    arrays = file_arrays(rows)
    selection = np.array([7, 0, 3, 3, 10])
    selected_rows = [rows[i] for i in selection]

    assert_arrays_equal(sparse_dots_npz.select_rows(arrays, selection), file_arrays(selected_rows), sparse_dots_npz.ARRAY_NAMES)

    expected = loaded_batch_arrays(selected_rows)
    batch = sparse_dots_npz.collate(arrays, selection)
    for name in expected:
        np.testing.assert_array_equal(batch[name], expected[name], err_msg=name)
    assert set(batch) == set(sparse_dots_npz.ROW_ARRAY_NAMES) | set(sparse_dots_npz.BATCH_PADDED_NAMES)


def test_collate_without_entities_keeps_length_one():
    rows = make_rows(3, 4)
    for row in rows:
        row["points"] = []
        row["pointOwnershipTargetsNP"] = []
    batch = sparse_dots_npz.collate(file_arrays(rows), np.arange(4))
    assert batch["pointCoordsNP"].shape == (4, 1, 2)
    assert not batch["pointCoordsNP"].any()


def test_group_by_length():
    rand = np.random.default_rng(0)
    point_counts = rand.integers(0, 1000, size=640)
    cand_counts = 1200 - point_counts
    rows = rand.permutation(640)
    batches = group_by_length(rows, point_counts, cand_counts, 16, 8, rand)
    assert len(batches) == 40
    assert all(len(b) == 16 for b in batches)
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(640))
    # A chunk of 8 batches is split by length: the batches of a chunk don't overlap in their point counts
    for start in range(0, 640, 128):
        chunk = set(rows[start:start + 128].tolist())
        ranges = sorted((point_counts[b].min(), point_counts[b].max()) for b in batches if set(b.tolist()) <= chunk)
        assert len(ranges) == 8
        assert all(ranges[i][1] <= ranges[i + 1][0] for i in range(7))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_loader_yields_every_row_once(tmp_path, num_workers):
    rows = make_rows(4, 70)
    filename = str(tmp_path / "data.npz")
    sparse_dots_npz.save_sparse_npz(filename, file_arrays(rows))
    batches = list(read_npz_training_data_sparse(
        [filename], batch_size=8, world_size=1, rank=0, device="cpu",
        randomize_symmetries=False, randomize_history=False, length_grouping_batches=4, num_workers=num_workers,
    ))
    assert len(batches) == 8
    seen = []
    for batch in batches:
        ids = batch["globalInputNC"][:, 0].long().tolist()
        seen.extend(ids)
        expected = loaded_batch_arrays([rows[i] for i in ids])
        for name in ["pointCountN", "pointCoordsNP", "pointDataNP", "pointChainNP", "candCountN", "candCoordsNM", "candDataNM",
                     "candPolicyTargetsNM", "candFutureposTargetsNM", "pointOwnershipTargetsNP", "groundN"]:
            np.testing.assert_array_equal(batch[name].numpy(), expected[name], err_msg=name)
        assert batch["pointDataNP"].dtype == torch.int32
        assert batch["candPolicyTargetsNM"].dtype == torch.float32
    assert len(set(seen)) == 64


def test_shuffle_keeps_every_row(tmp_path):
    rows = make_rows(5, 40)
    inputs = []
    for i in range(2):
        filename = str(tmp_path / f"in{i}.npz")
        inputs.append(filename)
        sparse_dots_npz.save_sparse_npz(filename, file_arrays(rows[:25] if i == 0 else rows[25:]))
    assert shuffle.is_sparse_npz_file(inputs[0]) and shuffle.is_sparse_npz_file(inputs[1])

    tmp_dirs = [str(tmp_path / f"tmp{j}") for j in range(2)]
    for d in tmp_dirs:
        os.makedirs(d)
    shuffle.shardify_sparse(0, inputs, 2, tmp_dirs, 1.0)
    outputs = []
    for j, d in enumerate(tmp_dirs):
        out_names = [str(tmp_path / f"out{j}_{k}.npz") for k in range(2)]
        outputs.extend(f for f, _ in shuffle.merge_bucket_sparse(out_names, 1, d))

    merged = sparse_dots_npz.concatenate([sparse_dots_npz.load_sparse_npz(f) for f in outputs])
    order = np.argsort(merged["globalInputNC"][:, 0])
    assert_arrays_equal(sparse_dots_npz.select_rows(merged, order), file_arrays(rows), sparse_dots_npz.ARRAY_NAMES)
