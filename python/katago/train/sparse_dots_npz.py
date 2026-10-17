# Files of the sparse Dots training data (model version 18), see docs/DotsSparseNN.md 7.2. Numpy only, shared by
# shuffle.py and the training data loader.
#
# Only facts are stored, no padding and no runs of zeros:
# - Per-row arrays have N rows.
# - Per-entity arrays (points, candidates, and the nonzero policy and futurepos targets of the candidates) are
#   concatenated in row order; the count arrays give the number of entities of each row.
# - The score distribution target is its (at most) two nonzero bins: their scores (bin minus the middle bin) and
#   their weights (summing to 100).
# - Small values are stored in int8.
#
# Files of older formats (without the chains of the points) are rejected.

from typing import Dict, List, Optional

import numpy as np

ROW_ARRAY_NAMES = [
    "boardSizeN", "globalInputNC", "pointCountN", "candCountN", "policyTargetCountN", "futureposTargetCountN",
    "groundN", "groundPolicyTargetsN", "globalTargetsNC", "scoreDistrScoresN", "scoreDistrWeightsN",
]
# The per-entity arrays of each count array
ENTITY_ARRAY_NAMES = {
    # owner, grounded, movesAgo, areaCount, areaMovesAgo (int8); insideMy, insideOpp (int16); the chain of the point
    # (int16, the 8-connected component of the points of its owner, numbered within the row)
    "pointCountN": ["pointCoordsP", "pointStateP", "pointInsideP", "pointChainP", "pointOwnershipTargetsP"],
    # gainMy, gainOpp (int16); inMyHouse, inOppHouse (int8)
    "candCountN": ["candCoordsM", "candGainM", "candHouseM"],
    # The candidates with a nonzero policy target (my move or the opponent's reply): their index and visits
    "policyTargetCountN": ["policyTargetCandT", "policyTargetsT"],
    # The candidates with a nonzero futurepos target (+8 or +32 moves): their index and the targets
    "futureposTargetCountN": ["futureposTargetCandF", "futureposTargetsF"],
}
ENTITY_COUNT_NAME = {name: count_name for count_name, names in ENTITY_ARRAY_NAMES.items() for name in names}
ARRAY_NAMES = ROW_ARRAY_NAMES + [name for names in ENTITY_ARRAY_NAMES.values() for name in names]

# Arrays of a batch built by collate(): the row arrays and these per-entity arrays padded to the max of the batch
BATCH_PADDED_NAMES = [
    "pointCoordsNP", "pointDataNP", "pointChainNP", "pointOwnershipTargetsNP",
    "candCoordsNM", "candDataNM", "candPolicyTargetsNM", "candFutureposTargetsNM",
]

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
        missing = [name for name in ARRAY_NAMES if name not in npz.files]
        if missing:
            raise ValueError(f"{filename} has an older format of the sparse Dots data (no {', '.join(missing)}), "
                             f"it must be generated again by the current katago")
        arrays = {name: npz[name] for name in ARRAY_NAMES}
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


def all_offsets(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """offsets() of every count array, to pass to collate()."""
    return {count_name: offsets(arrays[count_name]) for count_name in ENTITY_ARRAY_NAMES}


def select_rows(arrays: Dict[str, np.ndarray], rows: np.ndarray) -> Dict[str, np.ndarray]:
    """The selected rows (in the given order) with their entities."""
    rows = np.asarray(rows, dtype=np.int64)
    result = {name: arrays[name][rows] for name in ROW_ARRAY_NAMES}
    for count_name, names in ENTITY_ARRAY_NAMES.items():
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
    """A batch of the selected rows: the row arrays as they are, and the arrays of BATCH_PADDED_NAMES (the points and
    the candidates with their data and targets) padded with zeros to the max count of the batch (at least 1).
    starts: all_offsets(arrays), if precomputed."""
    rows = np.asarray(rows, dtype=np.int64)
    if starts is None:
        starts = all_offsets(arrays)
    batch = {name: arrays[name][rows] for name in ROW_ARRAY_NAMES}
    n = len(rows)

    def padded_entities(count_name):
        counts = arrays[count_name]
        src, dst_row, dst_pos = _entity_index(counts, starts[count_name], rows)
        length = max(1, int(counts[rows].max())) if n > 0 else 1
        return src, dst_row, dst_pos, length

    def scatter(length, dst_row, dst_pos, values, fill=0):
        out = np.full((n, length) + values.shape[1:], fill, dtype=values.dtype)
        out[dst_row, dst_pos] = values
        return out

    src, dst_row, dst_pos, num_points = padded_entities("pointCountN")
    batch["pointCoordsNP"] = scatter(num_points, dst_row, dst_pos, arrays["pointCoordsP"][src])
    point_data = np.concatenate([arrays["pointStateP"][src].astype(np.int16), arrays["pointInsideP"][src]], axis=1)
    batch["pointDataNP"] = scatter(num_points, dst_row, dst_pos, point_data)
    batch["pointChainNP"] = scatter(num_points, dst_row, dst_pos, arrays["pointChainP"][src], fill=-1)
    batch["pointOwnershipTargetsNP"] = scatter(num_points, dst_row, dst_pos, arrays["pointOwnershipTargetsP"][src])

    src, dst_row, dst_pos, num_cands = padded_entities("candCountN")
    batch["candCoordsNM"] = scatter(num_cands, dst_row, dst_pos, arrays["candCoordsM"][src])
    cand_data = np.concatenate([arrays["candGainM"][src], arrays["candHouseM"][src].astype(np.int16)], axis=1)
    batch["candDataNM"] = scatter(num_cands, dst_row, dst_pos, cand_data)

    # The sparse targets go to the candidates they name
    for count_name, cand_name, values_name, batch_name in (
        ("policyTargetCountN", "policyTargetCandT", "policyTargetsT", "candPolicyTargetsNM"),
        ("futureposTargetCountN", "futureposTargetCandF", "futureposTargetsF", "candFutureposTargetsNM"),
    ):
        src, dst_row, _ = _entity_index(arrays[count_name], starts[count_name], rows)
        batch[batch_name] = scatter(num_cands, dst_row, arrays[cand_name][src].astype(np.int64), arrays[values_name][src])
    return batch
