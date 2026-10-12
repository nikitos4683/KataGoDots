# Loading of the sparse Dots training data (model version 18), see docs/DotsSparseNN.md 7.2.
# Files are padded to their own maxima of points, areas, vertices and candidates; batches are cut down to the
# maxima of the batch. Symmetries only transform the coordinates.

import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import torch

from ..train.data_processing_pytorch import pad_global_targets_nc
from ..train.model_sparse_dots import NUM_MOVES_AGO, apply_symmetry_to_batch, trim_batch

# All arrays of a sparse data file, in the order they are written by the C++ trainingwrite
SPARSE_ARRAY_NAMES = [
    "boardSizeN", "globalInputNC",
    "pointCountN", "pointCoordsNP", "pointDataNP",
    "areaCountN", "areaDataNA",
    "vertexCountN", "vertexCoordsNV", "vertexAreaNV",
    "candCountN", "candCoordsNM", "candDataNM",
    "groundN",
    "candPolicyTargetsNM", "groundPolicyTargetsN", "candFutureposTargetsNM", "pointOwnershipTargetsNP",
    "globalTargetsNC", "scoreDistrN",
]

# Torch dtype of each array in a batch
_LONG_ARRAYS = {"pointCountN", "areaCountN", "vertexCountN", "candCountN", "vertexAreaNV"}
_INT_ARRAYS = {"boardSizeN", "pointCoordsNP", "pointDataNP", "areaDataNA", "vertexCoordsNV", "candCoordsNM", "candDataNM"}


def is_sparse_npz(npz) -> bool:
    return "pointCountN" in npz


def drop_history(batch: Dict[str, torch.Tensor], include_history: torch.Tensor) -> Dict[str, torch.Tensor]:
    """include_history [N,5] 0/1 with the KataGo semantics (once a move is dropped, all older ones are too).
    Removes the moves-ago marks of the points and areas older than the kept history."""
    kept = include_history.long().cumprod(dim=1).sum(dim=1)  # [N] number of kept moves
    result = dict(batch)
    point_data = batch["pointDataNP"].clone()
    moves_ago = point_data[..., 2]
    point_data[..., 2] = torch.where(moves_ago > kept.view(-1, 1), torch.zeros_like(moves_ago), moves_ago)
    result["pointDataNP"] = point_data
    area_data = batch["areaDataNA"].clone()
    moves_ago = area_data[..., 2]
    area_data[..., 2] = torch.where(moves_ago > kept.view(-1, 1), torch.zeros_like(moves_ago), moves_ago)
    result["areaDataNA"] = area_data
    return result


def random_symmetry(board_size: torch.Tensor, rand: np.random.Generator) -> int:
    """Any of the 8 symmetries if all the boards of the batch are square, otherwise one without transposition."""
    if bool(torch.all(board_size[:, 0] == board_size[:, 1])):
        return int(rand.integers(0, 8))
    return int(rand.integers(0, 4))


def read_npz_training_data_sparse(
    npz_files: List[str],
    batch_size: int,
    world_size: int,
    rank: int,
    device,
    randomize_symmetries: bool,
    randomize_history: bool = True,
    prefetch_depth: int = 1,
):
    rand = np.random.default_rng(seed=list(os.urandom(12)))

    def load_npz_file(npz_file):
        with np.load(npz_file) as npz:
            assert is_sparse_npz(npz), f"{npz_file} is not a sparse Dots data file"
            num_samples = npz["globalInputNC"].shape[0]
            num_whole_steps = num_samples // (batch_size * world_size)
            used = num_whole_steps * world_size * batch_size

            def select_rank_rows(arr):
                arr = arr[:used]
                rest = arr.shape[1:]
                arr = arr.reshape(num_whole_steps, world_size, batch_size, *rest)
                arr = arr[:, rank]
                return arr.reshape(num_whole_steps * batch_size, *rest)

            arrays = {name: select_rank_rows(npz[name]) for name in SPARSE_ARRAY_NAMES}
        arrays["globalTargetsNC"] = pad_global_targets_nc(arrays["globalTargetsNC"])
        return npz_file, num_whole_steps, arrays

    if not npz_files:
        return

    prefetch_depth = max(1, prefetch_depth)
    with ThreadPoolExecutor(max_workers=prefetch_depth) as executor:
        pending = deque()
        next_index = 0
        while next_index < len(npz_files) and len(pending) <= prefetch_depth:
            pending.append(executor.submit(load_npz_file, npz_files[next_index]))
            next_index += 1

        while pending:
            npz_file, num_whole_steps, arrays = pending.popleft().result()
            logging.info(f"Beginning {npz_file} with {num_whole_steps * world_size} usable batches, my rank is {rank}")
            if next_index < len(npz_files):
                pending.append(executor.submit(load_npz_file, npz_files[next_index]))
                next_index += 1

            for n in range(num_whole_steps):
                start = n * batch_size
                end = start + batch_size
                batch = {}
                for name, arr in arrays.items():
                    tensor = torch.from_numpy(np.ascontiguousarray(arr[start:end]))
                    if name in _LONG_ARRAYS:
                        tensor = tensor.long()
                    elif name in _INT_ARRAYS:
                        tensor = tensor.to(torch.int32)
                    else:
                        tensor = tensor.to(torch.float32)
                    batch[name] = tensor
                batch = trim_batch(batch)
                if randomize_history:
                    should_stop = torch.rand(batch_size, NUM_MOVES_AGO) >= 0.98
                    include_history = (torch.cumsum(should_stop, dim=1) <= 0).to(torch.float32)
                    batch = drop_history(batch, include_history)
                if randomize_symmetries:
                    batch = apply_symmetry_to_batch(batch, random_symmetry(batch["boardSizeN"], rand))
                yield {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
