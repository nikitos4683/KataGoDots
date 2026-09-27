# Selfplay Training in KataGoDots

KataGoDots includes a closed-loop self-play training system adapted for the game of Dots. This document describes the training pipeline, rectangular coordinate handling, and both single-machine synchronous and multi-machine asynchronous training setups.

---

## 1. Prerequisites

* **Compiled C++ engine**: Build KataGoDots with selfplay support (see [Compiling.md](Compiling.md)). Ensure `katago` executable is located in `cpp/` or on your path.
* **Python environment**: Python 3.8+ with PyTorch:
  ```bash
  pip install torch numpy
  ```
* **GPU**: A modern GPU for selfplay data generation and neural network training.

---

## 2. Training Loop Components

The selfplay loop consists of five components:

1. **Selfplay Engine** (`cpp/katago selfplay`):
   Simulates games of Dots using the latest neural net checkpoint in an accepted models directory, writing game records and training features to disk. Uses `cpp/configs/training/selfplay1_dots.cfg`.
2. **Shuffler** (`python/shuffle.py`, wrapped by `python/selfplay/shuffle.sh`):
   Reads raw selfplay records and shuffles them into `.npz` training batches with configurable lookback windows.
3. **Training** (`python/train.py`, wrapped by `python/selfplay/train.sh`):
   Trains neural network models on shuffled `.npz` chunks using PyTorch, stochastic weight averaging (SWA), and exports checkpoints to `torchmodels_toexport/`.
4. **Exporter** (`python/export_model_pytorch.py`, wrapped by `python/selfplay/export_model_for_selfplay.sh`):
   Converts PyTorch checkpoints into optimized `.bin.gz` models that the C++ engine loads for MCTS search.
5. **Gatekeeper** (`cpp/katago gatekeeper`):
   Matches candidate models against the current best model using `cpp/configs/training/gatekeeper1_dots.cfg`. If the new model wins by the configured margin, it is accepted as the new selfplay network. (Optional: gating can be disabled to accept all models directly).

---

## 3. Single-Machine Synchronous Loop

For training on a single machine or workstation, KataGoDots provides `python/selfplay/synchronous_loop.sh`. This script sequentially executes all five steps in each cycle:

```bash
./python/selfplay/synchronous_loop.sh <NAMEPREFIX> <BASEDIR> <TRAININGNAME> <MODELKIND> <USEGATING>
```

### Arguments:
* `NAMEPREFIX`: Globally unique prefix for exported models (e.g. `dotsrun`).
* `BASEDIR`: Root directory storing selfplay data, models, logs, and shuffle scratch space.
* `TRAININGNAME`: Sub-identifier for this training lineage (e.g. `run1`).
* `MODELKIND`: Model size configuration from `python/katago/train/modelconfigs.py` (e.g. `b6c96`, `b10c128`).
* `USEGATING`: `1` to test candidate models via the gatekeeper; `0` to accept every trained model directly.

### Example Run:
```bash
./python/selfplay/synchronous_loop.sh dots /workspace/dots_training run1 b6c96 0
```

### Cycle Configuration:
Key parameters can be customized at the top of `python/selfplay/synchronous_loop.sh`:
* `NUM_GAMES_PER_CYCLE`: Games played by selfplay per cycle (default: 400).
* `NUM_THREADS_FOR_SHUFFLING`: Shuffler CPU threads (default: 16).
* `BATCHSIZE`: Training batch size (default: 128).
* `NUM_TRAIN_SAMPLES_PER_EPOCH`: Samples per epoch chunk (default: 50,000).
* `SHUFFLE_MINROWS`: Minimum selfplay rows collected before starting training (default: 50,000).
* `SELFPLAY_CONFIG`: Points to `cpp/configs/training/selfplay1_dots.cfg`.
* `GATING_CONFIG`: Points to `cpp/configs/training/gatekeeper1_dots.cfg`.

The script queries `./bin/katago info` on launch to verify the compiled binary, extract `max_len_x` and `max_len_y` (39 and 32 by default), and verify Release build status.

---

## 4. Rectangular Coordinates & Data Representation

Unlike Go, Dots standard boards are rectangular (39x32). The training pipeline accounts for rectangular geometry:

* **Board Dimensions**: `train.py` accepts `--pos-len-x` and `--pos-len-y` (forwarded via `train.sh` as `$MAXLENX` and `$MAXLENY`).
* **Symmetry Augmentation**: Square boards allow 8 symmetries (rotations and reflections). Rectangular boards allow only the 4 rectangle-preserving symmetries (0: identity, 2: horizontal flip, 5: vertical flip, 7: 180-degree rotation). `katago/train/data_processing_pytorch.py` preserves rectangular coordinates and pass logits across these symmetries.
* **Input Features**: 22 spatial channels and 19 global channels as produced by `nninputsdots.cpp` and defined in `nninputs.h`.

---

## 5. Multi-Machine Asynchronous Training

For large-scale setups with multiple machines or separate GPUs:

1. **Shared Filesystem**: Set up `$BASEDIR` on a fast shared network filesystem accessible to all machines.
2. **Selfplay Workers**: Run one or more selfplay instances across GPU nodes:
   ```bash
   cpp/katago selfplay -output-dir $BASEDIR/selfplay -models-dir $BASEDIR/models -config cpp/configs/training/selfplay1_dots.cfg >> selfplay.log 2>&1 &
   ```
3. **Shuffler & Exporter**:
   ```bash
   cd python
   ./selfplay/shuffle_and_export_loop.sh $NAMEPREFIX $BASEDIR $SCRATCHDIR 16 128 0
   ```
4. **Training Process**:
   ```bash
   cd python
   ./selfplay/train.sh $BASEDIR $TRAININGNAME b6c96 128 main 39 32 $GITREV $BACKEND -lr-scale 1.0 -max-train-bucket-per-new-data 4 -no-repeat-files >> train.log 2>&1 &
   ```
5. **Gatekeeper (Optional)**:
   ```bash
   cpp/katago gatekeeper -rejected-models-dir $BASEDIR/rejectedmodels -accepted-models-dir $BASEDIR/models/ -sgf-output-dir $BASEDIR/gatekeepersgf/ -test-models-dir $BASEDIR/modelstobetested/ -config cpp/configs/training/gatekeeper1_dots.cfg >> gatekeeper.log 2>&1 &
   ```
