# Loading of the sparse Dots training data (model version 18), see docs/DotsSparseNN.md 7.2.
# Files are not padded (see sparse_dots_npz.py); a batch is padded to the maxima of its rows. To keep that padding
# small, the rows are grouped by length: chunks of a few batches are sorted by the number of points and the batches
# are played in random order. Symmetries only transform the coordinates. Batches are built in worker processes.

import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import torch

from ..train import sparse_dots_npz
from ..train.model_sparse_dots import NUM_MOVES_AGO, apply_symmetry_to_batch

# Torch dtype of each array in a batch
_LONG_ARRAYS = {"pointCountN", "candCountN"}
_INT_ARRAYS = {"boardSizeN", "pointCoordsNP", "pointDataNP", "candCoordsNM", "candDataNM"}

# Rows are sorted by length within chunks of this many batches. The longer the chunk, the less padding and the more
# alike the positions of a batch are (the same stage of the game): 32 batches leave ~5% padding of the tokens.
LENGTH_GROUPING_BATCHES = 32


def is_sparse_npz(npz) -> bool:
    return sparse_dots_npz.is_sparse_npz_names(npz.files)


def group_by_length(rows: np.ndarray, point_counts: np.ndarray, cand_counts: np.ndarray, batch_size: int,
                    chunk_batches: int, rand: np.random.Generator) -> List[np.ndarray]:
    """Splits rows (already in random order, a multiple of batch_size) into batches of rows of similar length.
    The number of points and the number of candidates are strongly anticorrelated (they share the board), so sorting
    by points makes both of them alike in a batch. Returns the batches in random order."""
    batches = []
    chunk_rows = batch_size * max(1, chunk_batches)
    for start in range(0, len(rows), chunk_rows):
        chunk = rows[start:start + chunk_rows]
        if chunk_batches > 1:
            chunk = chunk[np.lexsort((cand_counts[chunk], point_counts[chunk]))]
        batches.extend(chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size))
    return [batches[i] for i in rand.permutation(len(batches))]


def drop_history(batch: Dict[str, torch.Tensor], include_history: torch.Tensor) -> Dict[str, torch.Tensor]:
    """include_history [N,5] 0/1 with the KataGo semantics (once a move is dropped, all older ones are too).
    Removes the moves-ago marks of the points and of their areas older than the kept history.
    The freshest area mark of a point can't be recomputed then, a mark older than the kept history becomes 0."""
    kept = include_history.long().cumprod(dim=1).sum(dim=1).view(-1, 1)  # [N,1] number of kept moves
    result = dict(batch)
    point_data = batch["pointDataNP"].clone()
    for column in (2, 4):  # movesAgo of the dot, freshest movesAgo of its areas
        moves_ago = point_data[..., column]
        point_data[..., column] = torch.where(moves_ago > kept, torch.zeros_like(moves_ago), moves_ago)
    result["pointDataNP"] = point_data
    return result


def random_symmetry(board_size: torch.Tensor, rand: np.random.Generator) -> int:
    """Any of the 8 symmetries if all the boards of the batch are square, otherwise one without transposition."""
    if bool(torch.all(board_size[:, 0] == board_size[:, 1])):
        return int(rand.integers(0, 8))
    return int(rand.integers(0, 4))


# Batches are built in this many worker processes by default (0 = in the training process). Building a batch takes
# tens of milliseconds of CPU, as long as a training step, and threads would compete with the training for the GIL.
DEFAULT_NUM_WORKERS = int(os.environ.get("KATAGO_SPARSE_DOTS_DATA_WORKERS", "2"))


def _load_file_batches(npz_file: str, batch_size: int, world_size: int, rank: int, length_grouping_batches: int,
                       rand: np.random.Generator):
    arrays = sparse_dots_npz.load_sparse_npz(npz_file)
    num_samples = sparse_dots_npz.num_rows(arrays)
    num_whole_steps = num_samples // (batch_size * world_size)
    used = num_whole_steps * world_size * batch_size
    # The rows of this rank, the same split as the dense data
    rows = np.arange(used).reshape(num_whole_steps, world_size, batch_size)[:, rank].reshape(-1)
    batches = group_by_length(rows, arrays["pointCountN"], arrays["candCountN"], batch_size, length_grouping_batches, rand)
    starts = {name: sparse_dots_npz.offsets(arrays[name]) for name in ("pointCountN", "candCountN")}
    return npz_file, num_whole_steps, arrays, starts, batches


def _make_batch(arrays, starts, batch_rows, randomize_history: bool, randomize_symmetries: bool,
                rand: np.random.Generator) -> Dict[str, torch.Tensor]:
    """A batch of CPU tensors of the given rows, with the random history drop and symmetry."""
    batch = {}
    for name, arr in sparse_dots_npz.collate(arrays, batch_rows, starts).items():
        tensor = torch.from_numpy(arr)
        if name in _LONG_ARRAYS:
            tensor = tensor.long()
        elif name in _INT_ARRAYS:
            tensor = tensor.to(torch.int32)
        else:
            tensor = tensor.to(torch.float32)
        batch[name] = tensor
    if randomize_history:
        should_stop = torch.from_numpy(rand.random((len(batch_rows), NUM_MOVES_AGO)) >= 0.98)
        include_history = (torch.cumsum(should_stop, dim=1) <= 0).to(torch.float32)
        batch = drop_history(batch, include_history)
    if randomize_symmetries:
        batch = apply_symmetry_to_batch(batch, random_symmetry(batch["boardSizeN"], rand))
    return batch


def _file_batch_iter(npz_files: List[str], batch_size: int, world_size: int, rank: int, length_grouping_batches: int,
                     prefetch_depth: int, seed, part: int, num_parts: int):
    """For every file in order: (file, usable batches, arrays, starts, its batches of rows taken by this part).
    All the parts build the same batches (the grouping uses a generator seeded by the seed and the file index), each
    takes every num_parts-th of them. The next files are loaded in background threads."""
    def load(index):
        rand = np.random.default_rng([*seed, index])
        return _load_file_batches(npz_files[index], batch_size, world_size, rank, length_grouping_batches, rand)

    prefetch_depth = max(1, prefetch_depth)
    with ThreadPoolExecutor(max_workers=prefetch_depth) as executor:
        pending = deque()
        next_index = 0
        while next_index < len(npz_files) and len(pending) <= prefetch_depth:
            pending.append(executor.submit(load, next_index))
            next_index += 1
        while pending:
            npz_file, num_whole_steps, arrays, starts, batches = pending.popleft().result()
            if next_index < len(npz_files):
                pending.append(executor.submit(load, next_index))
                next_index += 1
            yield npz_file, num_whole_steps, arrays, starts, batches[part::num_parts]


class _SparseBatchDataset(torch.utils.data.IterableDataset):
    """The batches of the files, built in the DataLoader worker processes. Worker k builds the batches k, k + n, ...
    and the DataLoader takes the workers round-robin, so the order is the same as with a single process."""

    def __init__(self, npz_files, batch_size, world_size, rank, randomize_symmetries, randomize_history, prefetch_depth,
                 length_grouping_batches, seed):
        super().__init__()
        self.npz_files = list(npz_files)
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.randomize_symmetries = randomize_symmetries
        self.randomize_history = randomize_history
        self.prefetch_depth = prefetch_depth
        self.length_grouping_batches = length_grouping_batches
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        part, num_parts = (info.id, info.num_workers) if info is not None else (0, 1)
        rand = np.random.default_rng([*self.seed, len(self.npz_files), part])
        for _, _, arrays, starts, batches in _file_batch_iter(
            self.npz_files, self.batch_size, self.world_size, self.rank, self.length_grouping_batches,
            self.prefetch_depth, self.seed, part, num_parts,
        ):
            for batch_rows in batches:
                yield _make_batch(arrays, starts, batch_rows, self.randomize_history, self.randomize_symmetries, rand)


def read_npz_training_data_sparse(
    npz_files: List[str],
    batch_size: int,
    world_size: int,
    rank: int,
    device,
    randomize_symmetries: bool,
    randomize_history: bool = True,
    prefetch_depth: int = 1,
    length_grouping_batches: int = LENGTH_GROUPING_BATCHES,
    num_workers: int = None,
):
    """Yields the batches of the files as dicts of tensors on the device. With num_workers > 0 (by default
    DEFAULT_NUM_WORKERS) they are built in worker processes, otherwise in this one."""
    if not npz_files:
        return
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS
    seed = list(np.frombuffer(os.urandom(16), dtype=np.uint32))
    on_cuda = torch.device(device).type == "cuda"

    if num_workers <= 0:
        rand = np.random.default_rng([*seed, len(npz_files), 0])
        for npz_file, num_whole_steps, arrays, starts, batches in _file_batch_iter(
            npz_files, batch_size, world_size, rank, length_grouping_batches, prefetch_depth, seed, 0, 1,
        ):
            logging.info(f"Beginning {npz_file} with {num_whole_steps * world_size} usable batches, my rank is {rank}")
            for batch_rows in batches:
                batch = _make_batch(arrays, starts, batch_rows, randomize_history, randomize_symmetries, rand)
                yield {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
        return

    logging.info(f"Reading {len(npz_files)} sparse Dots data files with {num_workers} worker processes, my rank is {rank}")
    dataset = _SparseBatchDataset(
        npz_files, batch_size, world_size, rank, randomize_symmetries, randomize_history, prefetch_depth,
        length_grouping_batches, seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=num_workers, pin_memory=on_cuda, prefetch_factor=4,
    )
    for batch in loader:
        yield {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
