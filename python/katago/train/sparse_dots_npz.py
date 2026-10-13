# Files of the sparse Dots training data (model version 18), see docs/DotsSparseNN.md 7.2. Numpy only, shared by
# shuffle.py and the training data loader.
#
# Per-row arrays have N rows. The per-entity arrays are not padded: the points (candidates) of all the rows are
# concatenated in row order, pointCountN (candCountN) gives the number of each row. Older files had the per-entity
# arrays padded to the maxima of the file ([N, P, ...] named ...NP / ...NM); they are converted on loading.

from typing import Dict, List, Optional

import numpy as np

ROW_ARRAY_NAMES = [
    "boardSizeN", "globalInputNC", "pointCountN", "candCountN", "groundN",
    "groundPolicyTargetsN", "globalTargetsNC", "scoreDistrN",
]
POINT_ARRAY_NAMES = ["pointCoordsP", "pointDataP", "pointOwnershipTargetsP"]
CAND_ARRAY_NAMES = ["candCoordsM", "candDataM", "candPolicyTargetsM", "candFutureposTargetsM"]
ARRAY_NAMES = ROW_ARRAY_NAMES + POINT_ARRAY_NAMES + CAND_ARRAY_NAMES

# Name of each per-entity array in a padded batch (and in the older padded files)
PADDED_NAME = {
    "pointCoordsP": "pointCoordsNP", "pointDataP": "pointDataNP", "pointOwnershipTargetsP": "pointOwnershipTargetsNP",
    "candCoordsM": "candCoordsNM", "candDataM": "candDataNM", "candPolicyTargetsM": "candPolicyTargetsNM",
    "candFutureposTargetsM": "candFutureposTargetsNM",
}
COUNT_NAME = {name: "pointCountN" for name in POINT_ARRAY_NAMES}
COUNT_NAME.update({name: "candCountN" for name in CAND_ARRAY_NAMES})

GLOBAL_TARGETS_NC_CHANNELS = 80


def is_sparse_npz_names(names) -> bool:
    return "pointCountN" in names or "pointCountN.npy" in names


def pad_global_targets_nc(global_targets: np.ndarray) -> np.ndarray:
    """Zero-pads older-format globalTargetsNC rows up to the current channel count."""
    num_channels = global_targets.shape[1]
    if num_channels == GLOBAL_TARGETS_NC_CHANNELS:
        return global_targets
    assert num_channels < GLOBAL_TARGETS_NC_CHANNELS, f"globalTargetsNC has {num_channels} channels, more than {GLOBAL_TARGETS_NC_CHANNELS}"
    padded = np.zeros((global_targets.shape[0], GLOBAL_TARGETS_NC_CHANNELS), dtype=global_targets.dtype)
    padded[:, :num_channels] = global_targets
    return padded


def num_rows(arrays: Dict[str, np.ndarray]) -> int:
    return arrays["globalInputNC"].shape[0]


def offsets(counts: np.ndarray) -> np.ndarray:
    """Start of each row in the concatenated per-entity arrays."""
    counts = counts.astype(np.int64)
    return np.cumsum(counts) - counts


def load_sparse_npz(filename: str) -> Dict[str, np.ndarray]:
    with np.load(filename) as npz:
        assert is_sparse_npz_names(npz.files), f"{filename} is not a sparse Dots data file"
        arrays = {name: npz[name] for name in ROW_ARRAY_NAMES}
        for name, padded_name in PADDED_NAME.items():
            if name in npz.files:
                arrays[name] = npz[name]
            else:
                # Older padded file: keep the entries of each row below its count, in row order
                padded = npz[padded_name]
                counts = arrays[COUNT_NAME[name]].astype(np.int64)
                arrays[name] = padded[np.arange(padded.shape[1])[None, :] < counts[:, None]]
    arrays["globalTargetsNC"] = pad_global_targets_nc(arrays["globalTargetsNC"])
    return arrays


def _entity_index(counts: np.ndarray, starts: np.ndarray, rows: np.ndarray):
    """For the selected rows: indices of their entities in the concatenated arrays, the row of each entity in the
    selection and its index within the row."""
    row_counts = counts[rows].astype(np.int64)
    total = int(row_counts.sum())
    out_starts = np.cumsum(row_counts) - row_counts
    within = np.arange(total, dtype=np.int64) - np.repeat(out_starts, row_counts)
    return np.repeat(starts[rows], row_counts) + within, np.repeat(np.arange(len(rows)), row_counts), within


def select_rows(arrays: Dict[str, np.ndarray], rows: np.ndarray) -> Dict[str, np.ndarray]:
    """The selected rows (in the given order) with their entities."""
    rows = np.asarray(rows, dtype=np.int64)
    result = {name: arrays[name][rows] for name in ROW_ARRAY_NAMES}
    for count_name, names in (("pointCountN", POINT_ARRAY_NAMES), ("candCountN", CAND_ARRAY_NAMES)):
        counts = arrays[count_name]
        src, _, _ = _entity_index(counts, offsets(counts), rows)
        for name in names:
            result[name] = arrays[name][src]
    return result


def concatenate(arrays_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if len(arrays_list) == 1:
        return arrays_list[0]
    return {name: np.concatenate([a[name] for a in arrays_list], axis=0) for name in ARRAY_NAMES}


def save_sparse_npz(filename: str, arrays: Dict[str, np.ndarray], rows: Optional[np.ndarray] = None):
    if rows is not None:
        arrays = select_rows(arrays, rows)
    np.savez_compressed(filename, **{name: arrays[name] for name in ARRAY_NAMES})


def collate(arrays: Dict[str, np.ndarray], rows: np.ndarray, starts: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """A batch of the selected rows: the per-row arrays as they are, the per-entity arrays padded with zeros to the
    maximum count of the batch (at least 1), under their padded names. starts: offsets() of pointCountN and
    candCountN, if precomputed."""
    rows = np.asarray(rows, dtype=np.int64)
    batch = {name: arrays[name][rows] for name in ROW_ARRAY_NAMES}
    for count_name, names in (("pointCountN", POINT_ARRAY_NAMES), ("candCountN", CAND_ARRAY_NAMES)):
        counts = arrays[count_name]
        count_starts = starts[count_name] if starts is not None else offsets(counts)
        src, dst_row, dst_pos = _entity_index(counts, count_starts, rows)
        length = max(1, int(counts[rows].max())) if len(rows) > 0 else 1
        for name in names:
            flat = arrays[name]
            padded = np.zeros((len(rows), length) + flat.shape[1:], dtype=flat.dtype)
            padded[dst_row, dst_pos] = flat[src]
            batch[PADDED_NAME[name]] = padded
    return batch
