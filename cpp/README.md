# C++ Source Code Overview

This document summarizes the C++ source architecture of **KataGoDots**, in approximate dependency order from lowest level to highest.

---

## 1. Source Directories

### `core/`
Low-level utilities layered over the C++ standard library:
* High-performance hashing (MurmurHash, Zobrist hashing).
* Portable random number generation (`rand.{cpp,h}`).
* String formatting, command-line parsing, config file reading (`config_parser.{cpp,h}`).
* Multithreading and queue utilities (`threadsafequeue.h`).

### `game/`
Board representations, Dots mechanics, and rulesets:
* `board.{cpp,h}`: Core `Board` representation supporting both Go and Dots. Implements rectangular coordinate systems (`x_size`, `y_size`) with stride indexing `(x + 1) + (y + 1) * (x_size + 1)`.
* `dotsfield.cpp`: Core Dots algorithms for tracing cycles, detecting enclosed bases, and setting territory flags.
* `dotsfieldCapturesAndTerritories.cpp`: Capture and base evaluations, computing captured dot locations and empty base areas.
* `dotsfieldLadders.{cpp,h}`: Tactical ladder detection and `DotsLaddersSolver`, solving attacking/defending ladder paths and identifying working moves.
* `dotsboardhistory.cpp`: Move history, grounding alive checks (`winOrEffectiveDrawByGrounding`), resign reasonableness (`isResignReasonableForDots`), and Dots scoring.
* `rules.{cpp,h}`: Rules struct, start position generators (`START_POS_EMPTY`, `START_POS_SINGLE`, `START_POS_CROSS`, `START_POS_CROSS_2`, `START_POS_CROSS_4`), and preset definitions (`bbs`, `notago`, `russian`).
* `boardhistory.{cpp,h}`: Move history tracking, superko detection, and pass/grounding coordination.

### `neuralnet/`
Inference server and backend implementations:
* `nninputs.{cpp,h}`: Definitions of spatial and global neural net input features.
* `nninputsdots.cpp`: Dots feature extraction producing 22 spatial planes (including tactical ladder planes 14–17) and 19 global planes.
* `nneval.{cpp,h}`: Thread-safe multi-threaded batched neural net evaluation server.
* `desc.{cpp,h}`: Model architecture descriptors and `.bin.gz` weight loading.
* `modelversion.{cpp,h}`: Model version enumerations.
* `onnxmodelbuilder.{cpp,h}`: In-memory ONNX graph construction for TensorRT and ONNX backends.
* **Backends**:
  * `openclbackend.cpp`: General OpenCL GPU backend.
  * `cudabackend.cpp`: NVIDIA CUDA + cuDNN backend.
  * `trtbackend.cpp`: NVIDIA TensorRT backend (via ONNX parser).
  * `rocmbackend.cpp`: AMD ROCm / MIOpen backend.
  * `eigenbackend.cpp`: CPU backend with optional AVX2/FMA vectorization.
  * `onnxbackend.cpp`: Cross-platform ONNX Runtime backend.
  * `metalbackend.{cpp,h}` / `metalbackend.swift`: Apple Silicon Metal / CoreML backend.

### `search/`
Monte-Carlo Tree Search (MCTS) engine:
* `search.{cpp,h}`: Multithreaded MCTS implementation with support for Monte-Carlo Graph Search (MCGS), terminal grounded state detection, and Dots-normalized move temperature decay.
* `searchparams.{cpp,h}`: Configurable search hyperparameters.
* `timecontrols.{cpp,h}`: Time control management, including Bronstein delay for Dots (`time_settings <main> <per_move>`).
* `searchresults.cpp`: Move selection and statistics reporting.

### `dataio/`
File serialization and dataset generation:
* `sgf.{cpp,h}`: SGF parser and writer supporting Dots board sizes, moves, and start position presets.
* `trainingwrite.{cpp,h}`: Selfplay training data writer for `.npz` records.
* `loadmodel.{cpp,h}`: Model weight loader from disk.

### `command/`
CLI subcommands:
* `gtp.cpp`: GTP engine with custom extensions (`get_boardsize`, `get_moves`, `get_position`, multi-move `play`, `undo [count]`, multi-move `genmove [color] [moves_count]`, `info`, `time_settings`, `time_left`).
* `analysis.cpp`: JSON parallel analysis engine with Dots query/response support (`dots`, `playerToMove`, `initialStones`, `timeControl`, `resignReasonable`, `chosenMove`).
* `selfplay.cpp`: High-throughput selfplay data generator.
* `gatekeeper.cpp`: Candidate network evaluation against current best model.
* `match.cpp`: Multi-model tournament and benchmark matches.
* `benchmark.cpp`: Hardware speed benchmark and search thread recommendations.
* `dumponnx.cpp`: ONNX graph exporter from `.bin.gz` models.
* `runtests.cpp`: Comprehensive unit, stress, and regression test runner.

### `tests/`
Test suites covering engine logic and game rules:
* `testdotsbasic.cpp`: Dots territory, capture, and base boundary tests.
* `testdotsextra.cpp`: Edge-case capture scenarios and cycle detections.
* `testdotsladders.cpp`: Tactical ladder detection and solver tests.
* `testdotsstartposes.cpp`: Start position generator and recognizer tests.
* `testdotsstress.cpp`: High-volume random Dots simulation stress tests (100k+ games).
* `testdotsutils.{cpp,h}`: Helper assertions for Dots test positions.
* `testboardbasic.cpp`, `testboardarea.cpp`, `testrules.cpp`, `testtime.cpp`: General board, rules, and Bronstein time control tests.

---

## 2. Configuration Files (`cpp/configs/`)

* `gtp_dots.cfg`: GTP engine configuration for Dots games.
* `analysis_dots.cfg`: Parallel JSON analysis engine configuration for Dots.
* `match_dots.cfg`: Match configuration for comparing Dots models.
* `training/selfplay1_dots.cfg`: Selfplay data generation configuration for Dots (39x32 board).
* `training/gatekeeper1_dots.cfg`: Gatekeeper testing configuration for Dots.
