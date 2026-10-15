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
# Older files are converted on loading: the unpadded files with dense candidate targets, a full score distribution
# and int16 data, and before them the files with the per-entity arrays padded to the maxima of the file.

from typing import Dict, List, Optional

import numpy as np

ROW_ARRAY_NAMES = [
    "boardSizeN", "globalInputNC", "pointCountN", "candCountN", "policyTargetCountN", "futureposTargetCountN",
    "groundN", "groundPolicyTargetsN", "globalTargetsNC", "scoreDistrScoresN", "scoreDistrWeightsN",
]
# The per-entity arrays of each count array
ENTITY_ARRAY_NAMES = {
    # owner, grounded, movesAgo, areaCount, areaMovesAgo (int8); insideMy, insideOpp (int16)
    "pointCountN": ["pointCoordsP", "pointStateP", "pointInsideP", "pointOwnershipTargetsP"],
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
    "pointCoordsNP", "pointDataNP", "pointOwnershipTargetsNP",
    "candCoordsNM", "candDataNM", "candPolicyTargetsNM", "candFutureposTargetsNM",
]

GLOBAL_TARGETS_NC_CHANNELS = 80

# Older formats: the unpadded one with dense candidate targets, and the names of its arrays padded to the maxima
# of the file in the oldest one
_V1_POINT_ARRAY_NAMES = ["pointCoordsP", "pointDataP", "pointOwnershipTargetsP"]
_V1_CAND_ARRAY_NAMES = ["candCoordsM", "candDataM", "candPolicyTargetsM", "candFutureposTargetsM"]
_V1_PADDED_NAME = {
    "pointCoordsP": "pointCoordsNP", "pointDataP": "pointDataNP", "pointOwnershipTargetsP": "pointOwnershipTargetsNP",
    "candCoordsM": "candCoordsNM", "candDataM": "candDataNM", "candPolicyTargetsM": "candPolicyTargetsNM",
    "candFutureposTargetsM": "candFutureposTargetsNM",
}


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


def score_distribution_bins(dense: np.ndarray):
    """A full score distribution [N, L] (one or two adjacent nonzero bins per row, the middle bin at L / 2) as the
    scores of two adjacent bins [N, 2] int16 (ascending; bin minus the middle bin) and their weights [N, 2] int8.
    A single nonzero bin is paired with the next one (the previous one at the end) with weight 0, like
    TrainingWriteBuffers::writeDotsSparseArrays does."""
    length = dense.shape[1]
    nonzero = dense != 0
    counts = nonzero.sum(axis=1)
    assert counts.min(initial=1) >= 1 and counts.max(initial=1) <= 2, "A score distribution row needs one or two nonzero bins"
    first_two = np.argsort(~nonzero, axis=1, kind="stable")[:, :2]
    lo, hi = first_two[:, 0], first_two[:, 1]
    single = counts == 1
    hi = np.where(single, lo + 1, hi)
    at_end = single & (lo == length - 1)
    lo = np.where(at_end, length - 2, lo)
    hi = np.where(at_end, length - 1, hi)
    bins = np.stack([lo, hi], axis=1)
    assert ((bins[:, 1] - bins[:, 0]) == 1).all(), "The nonzero bins of a score distribution row must be adjacent"
    weights = np.take_along_axis(dense, bins, axis=1)
    return (bins - length // 2).astype(np.int16), weights.astype(np.int8)


def _row_of_entities(counts: np.ndarray):
    """For the concatenated entities of the rows: the row of each one and its index within the row."""
    counts = counts.astype(np.int64)
    rows = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    within = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(offsets(counts), counts)
    return rows, within


def _sparse_targets(dense: np.ndarray, cand_counts: np.ndarray):
    """Dense per-candidate targets [ΣM, 2] as the count of nonzero candidates of each row, their indices and values."""
    rows, within = _row_of_entities(cand_counts)
    nonzero = (dense != 0).any(axis=1)
    counts = np.bincount(rows[nonzero], minlength=len(cand_counts)).astype(np.int16)
    return counts, within[nonzero].astype(np.int16), dense[nonzero]


def _convert_v1(v1: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The arrays of an unpadded file with dense candidate targets (pointDataP [ΣP, 7] int16, candDataM [ΣM, 4]
    int16, candPolicyTargetsM, candFutureposTargetsM, scoreDistrN) in the current format."""
    arrays = {name: v1[name] for name in ["boardSizeN", "globalInputNC", "pointCountN", "candCountN", "groundN",
                                          "groundPolicyTargetsN", "globalTargetsNC"]}
    point_data = v1["pointDataP"]
    arrays["pointCoordsP"] = v1["pointCoordsP"]
    arrays["pointStateP"] = point_data[:, :5].astype(np.int8)
    arrays["pointInsideP"] = point_data[:, 5:7].astype(np.int16)
    arrays["pointOwnershipTargetsP"] = v1["pointOwnershipTargetsP"]
    cand_data = v1["candDataM"]
    arrays["candCoordsM"] = v1["candCoordsM"]
    arrays["candGainM"] = cand_data[:, :2].astype(np.int16)
    arrays["candHouseM"] = cand_data[:, 2:4].astype(np.int8)
    cand_counts = v1["candCountN"]
    arrays["policyTargetCountN"], arrays["policyTargetCandT"], policy = _sparse_targets(v1["candPolicyTargetsM"], cand_counts)
    arrays["policyTargetsT"] = policy.astype(np.int16)
    arrays["futureposTargetCountN"], arrays["futureposTargetCandF"], futurepos = _sparse_targets(v1["candFutureposTargetsM"], cand_counts)
    arrays["futureposTargetsF"] = futurepos.astype(np.int8)
    arrays["scoreDistrScoresN"], arrays["scoreDistrWeightsN"] = score_distribution_bins(v1["scoreDistrN"])
    return arrays


def load_sparse_npz(filename: str) -> Dict[str, np.ndarray]:
    with np.load(filename) as npz:
        assert is_sparse_npz_names(npz.files), f"{filename} is not a sparse Dots data file"
        if "pointStateP" in npz.files:
            arrays = {name: npz[name] for name in ARRAY_NAMES}
        else:
            v1 = {name: npz[name] for name in ["boardSizeN", "globalInputNC", "pointCountN", "candCountN", "groundN",
                                               "groundPolicyTargetsN", "globalTargetsNC", "scoreDistrN"]}
            for name, padded_name in _V1_PADDED_NAME.items():
                if name in npz.files:
                    v1[name] = npz[name]
                else:
                    # Oldest padded file: keep the entries of each row below its count, in row order
                    padded = npz[padded_name]
                    count_name = "pointCountN" if name in _V1_POINT_ARRAY_NAMES else "candCountN"
                    counts = v1[count_name].astype(np.int64)
                    v1[name] = padded[np.arange(padded.shape[1])[None, :] < counts[:, None]]
            arrays = _convert_v1(v1)
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

    def scatter(length, dst_row, dst_pos, values):
        out = np.zeros((n, length) + values.shape[1:], dtype=values.dtype)
        out[dst_row, dst_pos] = values
        return out

    src, dst_row, dst_pos, num_points = padded_entities("pointCountN")
    batch["pointCoordsNP"] = scatter(num_points, dst_row, dst_pos, arrays["pointCoordsP"][src])
    point_data = np.concatenate([arrays["pointStateP"][src].astype(np.int16), arrays["pointInsideP"][src]], axis=1)
    batch["pointDataNP"] = scatter(num_points, dst_row, dst_pos, point_data)
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
