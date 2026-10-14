"""Files of the sparse Dots training data: the unpadded format, older padded files, shuffling and batching."""

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


def padded_arrays(rows):
    """The rows as numpy arrays with the per-entity arrays padded to the maxima (the older file format)."""
    batch = {name: tensor.numpy() for name, tensor in collate(rows, score_len=SCORE_LEN).items()}
    batch["scoreDistrN"] = batch["scoreDistrN"].astype(np.int8)
    return batch


def file_arrays(rows):
    """The rows as the arrays of an unpadded file."""
    padded = padded_arrays(rows)
    arrays = {name: padded[name] for name in sparse_dots_npz.ROW_ARRAY_NAMES}
    for name, padded_name in sparse_dots_npz.PADDED_NAME.items():
        counts = padded[sparse_dots_npz.COUNT_NAME[name]]
        arrays[name] = np.concatenate([padded[padded_name][i, :counts[i]] for i in range(len(rows))], axis=0)
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
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_load_unpadded_and_older_padded_files(tmp_path):
    rows = make_rows(1, 13)
    expected = file_arrays(rows)
    expected["globalTargetsNC"] = sparse_dots_npz.pad_global_targets_nc(expected["globalTargetsNC"])

    unpadded = str(tmp_path / "unpadded.npz")
    sparse_dots_npz.save_sparse_npz(unpadded, file_arrays(rows))
    assert_arrays_equal(sparse_dots_npz.load_sparse_npz(unpadded), expected, sparse_dots_npz.ARRAY_NAMES)

    older = str(tmp_path / "older.npz")
    padded = padded_arrays(rows)
    np.savez_compressed(older, **padded)
    assert_arrays_equal(sparse_dots_npz.load_sparse_npz(older), expected, sparse_dots_npz.ARRAY_NAMES)


def test_select_rows_and_collate():
    rows = make_rows(2, 11)
    arrays = file_arrays(rows)
    selection = np.array([7, 0, 3, 3, 10])
    selected_rows = [rows[i] for i in selection]

    assert_arrays_equal(sparse_dots_npz.select_rows(arrays, selection), file_arrays(selected_rows), sparse_dots_npz.ARRAY_NAMES)

    expected = padded_arrays(selected_rows)
    batch = sparse_dots_npz.collate(arrays, selection)
    assert_arrays_equal(batch, expected, expected.keys())
    assert set(batch.keys()) == set(expected.keys())


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
        expected = padded_arrays([rows[i] for i in ids])
        for name in ["pointCountN", "pointCoordsNP", "pointDataNP", "candCountN", "candCoordsNM", "candDataNM",
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
        if i == 0:
            sparse_dots_npz.save_sparse_npz(filename, file_arrays(rows[:25]))
        else:
            np.savez_compressed(filename, **padded_arrays(rows[25:]))  # an older padded file
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
    expected = file_arrays(rows)
    expected["globalTargetsNC"] = sparse_dots_npz.pad_global_targets_nc(expected["globalTargetsNC"])
    assert_arrays_equal(sparse_dots_npz.select_rows(merged, order), expected, sparse_dots_npz.ARRAY_NAMES)
