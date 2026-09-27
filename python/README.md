# Python Source Code Overview

This directory contains the Python training pipeline, data processing, and model export infrastructure for **KataGoDots**.

---

## 1. Main Training Pipeline

* `shuffle.py`: Multi-threaded selfplay data shuffler. Collects raw game records from `selfplay/` and formats them into `.npz` training batches with configurable lookback windows. Wrapped by `selfplay/shuffle.sh`.
* `train.py`: Main PyTorch training entry point. Loads shuffled batches, trains the neural network with weight decay and SWA (stochastic weight averaging), and saves checkpoints. Accepts `--pos-len-x` and `--pos-len-y` for rectangular Dots boards. Wrapped by `selfplay/train.sh`.
* `export_model_pytorch.py`: Converts PyTorch checkpoints (`.pt`) to KataGo's optimized binary format (`.bin.gz`) for loading into the C++ engine. Wrapped by `selfplay/export_model_for_selfplay.sh`.
* `katago/train/model_pytorch.py`: Neural network definitions including trunk (convnet and transformer blocks), policy heads, and value heads. Supports `Game.DOTS`, 22 spatial input features, 19 global features, and rectangular board dimensions (`pos_len_x`, `pos_len_y`).
* `katago/train/data_processing_pytorch.py`: Data loading and augmentation for `.npz` files. Supports rectangular board geometries and restricts symmetry transformations to the 4 rectangle-preserving symmetries (0: identity, 2: horizontal flip, 5: vertical flip, 7: 180-degree rotation).
* `katago/train/metrics_pytorch.py`: Loss functions for policy, value, score distribution, and auxiliary targets.
* `katago/train/modelconfigs.py`: Architecture specifications for model sizes (e.g. `b6c96`, `b10c128`, `b15c192`).

---

## 2. Automation Scripts (`selfplay/`)

* `synchronous_loop.sh`: Single-machine training loop that runs gatekeeper, selfplay, shuffle, train, and export sequentially. Recommended for local training and experimentation.
  ```bash
  ./selfplay/synchronous_loop.sh <NAMEPREFIX> <BASEDIR> <TRAININGNAME> <MODELKIND> <USEGATING>
  ```
* `train.sh`: Wrapper for `train.py`, handling parameter forwarding (`$MAXLENX`, `$MAXLENY`, batch size, learning rates) and script snapshotting.
* `shuffle.sh`: Wrapper for `shuffle.py`, managing temporary scratch directories and lookback windows.
* `export_model_for_selfplay.sh`: Exports checkpoints from `torchmodels_toexport/` to `models/` or `modelstobetested/`.
* `shuffle_and_export_loop.sh`: Continuous background loop running shuffler and exporter for asynchronous cluster setups.

---

## 3. Tests (`tests/`)

Unit tests can be run using `pytest`:
```bash
pytest
```

Key test suites:
* `tests/test_rectangular_training_data.py`: Verifies that `.npz` data loader preserves rectangular coordinates, pass moves, and rectangle-preserving symmetries.
* `tests/test_training_data_generator.py`: Tests the shuffler data pipeline and windowing behavior.
* `tests/test_floored_weight_decay.py`: Tests optimizer behavior.
