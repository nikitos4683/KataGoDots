# AGENTS.md — Developer and AI Agent Guide for KataGoDots

Welcome to **KataGoDots**! This document provides technical context, architecture maps, build and test instructions, and operational guidelines for AI coding agents and human developers working on this codebase.

---

## 1. Project Overview

**KataGoDots** is an adaptation and fork of [KataGo](https://github.com/lightvector/KataGo) (v1.17+) engineered specifically for the game of **Dots** (Russian: *Точки* / *Игра в точки*), while maintaining underlying support for Go.

- **Primary Maintainers**: Ivan Kochurkin ([@kvanttt](https://github.com/kvanttt)), Nikita Shokarev ([@nikitos4683](https://github.com/nikitos4683)).
- **Core Technology Stack**:
  - **C++17**: High-performance MCTS/MCGS search engine, Dots cycle/territory tracing, board simulation, multi-backend GPU/CPU neural network inference server (`cpp/`).
  - **Python 3.10+ / PyTorch**: Neural network training pipeline, data shuffling, self-play orchestration, model export (`python/`).
  - **CMake (>= 3.18.2)**: Cross-platform build system supporting Windows (MSVC, MinGW), Linux (GCC, Clang), and macOS (AppleClang, Swift).

---

## 2. Game Mechanics: Dots vs. Go

Understanding the differences between Go and Dots is vital when editing game logic, features, or heuristics:

| Feature | Go | Dots (*Точки*) |
| :--- | :--- | :--- |
| **Grid / Board Size** | Square (typically 19x19, 13x13, 9x9) | Rectangular (standard default **39x32**; configurable e.g. 20x20) |
| **Move Placement** | Stones on intersections | Dots on intersections |
| **Captures / Territory** | Surrounding stones remove them from board | Surrounding opponent dots creates an enclosure (**base / база**). Enclosed dots remain on board as captured points |
| **Pass / Grounding** | Pass move; two consecutive passes end game | Pass (`PASS_LOC`) represents **grounding (заземление)** — connecting a base to the border or grounded area. Game ends if a player grounds alive or position is terminal |
| **Start Positions** | Usually empty board (or handicap stones) | Presets: `EMPTY` (0), `SINGLE` (1), `CROSS` (2, 1 скрест), `CROSS_2` (3), `CROSS_4` (4, 4 скреста), with optional random perturbation |
| **Standard Rulesets** | Japanese, Chinese, Tromp-Taylor, etc. | `bbs` (standard single cross), `notago` (4 random crosses) |
| **Time Controls** | Byo-yomi, Canadian, Fischer | **Bronstein delay** (`time_settings <main> <per-move>`): each move uses per-move delay first; unused delay does not bank |
| **Expected Game Length** | ~250 moves on 19x19 (~0.69 moves/point) | ~400 moves on 39x32 (~0.32 moves/point). Move temperature decay is normalized against Dots expected length |
| **Auxiliary Heads** | Seki heads, score distribution heads | Obsolete seki/scoring heads removed. Ownership prediction is masked to placed dots only |

---

## 3. Codebase Architecture

```
KataGoDots/
├── cpp/                           # C++ engine, search, and inference
│   ├── book/                      # Opening book utilities
│   ├── command/                   # CLI subcommands (gtp, analysis, selfplay, runtests, benchmark, gatekeeper)
│   ├── configs/                   # Engine configurations
│   │   ├── analysis_dots.cfg      # Analysis engine config for Dots
│   │   ├── gtp_dots.cfg           # GTP engine config for Dots
│   │   ├── match_dots.cfg         # Match config for Dots
│   │   └── training/              # Training configs (selfplay1_dots.cfg, gatekeeper1_dots.cfg, etc.)
│   ├── core/                      # Utility layer (hashing, threading, queues, config parser, RNG)
│   ├── dataio/                    # SGF parser/writer, numpy output, model loading
│   ├── distributed/               # Distributed training client
│   ├── game/                      # Board representation and game rules
│   │   ├── board.{h,cpp}          # Core Board representation (supports both Go and Dots)
│   │   ├── boardhistory.{h,cpp}   # Move history, superko, score evaluation
│   │   ├── common.h               # Core types (Loc, Color, Player, Move, string keys)
│   │   ├── dotsfield.cpp          # Dots cycle tracing, base boundaries, territory flags
│   │   ├── dotsfieldCapturesAndTerritories.cpp # Capture/territory calculation
│   │   ├── dotsfieldLadders.{h,cpp} # Dots tactical ladder solver
│   │   ├── dotsboardhistory.cpp   # Dots scoring, grounding alive checks, resign reasonableness
│   │   └── rules.{h,cpp}          # Rules struct, start position generators/recognizers
│   ├── neuralnet/                 # Neural net backends & feature extractors
│   │   ├── desc.{h,cpp}           # Model descriptor & weight parsing
│   │   ├── modelversion.{h,cpp}   # Model version enumerations
│   │   ├── nneval.{h,cpp}         # Multi-threaded batched evaluation server
│   │   ├── nninputs.{h,cpp}       # Feature extraction interfaces
│   │   ├── nninputsdots.cpp       # Dots spatial (22) and global (19) input features
│   │   ├── cudabackend.cpp        # CUDA + cuDNN backend
│   │   ├── trtbackend.cpp         # TensorRT 10 & 11 backend (with onnxmodelbuilder)
│   │   ├── openclbackend.cpp      # OpenCL backend
│   │   └── eigenbackend.cpp       # CPU backend with AVX2/FMA
│   ├── program/                   # Top-level gameplay, setup, GTP config parsing
│   ├── search/                    # MCTS/MCGS search engine
│   │   ├── search.{h,cpp}         # Multithreaded MCTS search implementation
│   │   ├── timecontrols.{h,cpp}   # Time controls including Bronstein delay
│   │   └── searchparams.{h,cpp}   # Search parameters and coefficients
│   ├── tests/                     # C++ unit, stress, and regression tests
│   └── CMakeLists.txt             # Primary C++ CMake build file
├── python/                        # Python training pipeline and model definitions
│   ├── katago/
│   │   ├── train/
│   │   │   ├── model_pytorch.py   # PyTorch model architecture (trunk, policy, value, ownership)
│   │   │   ├── metrics_pytorch.py # Loss functions and training metrics
│   │   │   ├── modelconfigs.py    # Model size configurations (e.g. b6c96, b10c128)
│   │   │   └── data_processing_pytorch.py # Training data loading and augmentation
│   │   └── utils/
│   │       └── training_data_generator.py # Training data file cycling and gap-delay shuffler
│   ├── selfplay/                  # Selfplay automation scripts
│   │   ├── synchronous_loop.sh    # Single-machine synchronous loop: gatekeeper -> selfplay -> shuffle -> train -> export
│   │   ├── train.sh               # Python training script wrapper
│   │   ├── shuffle.sh             # Data shuffler script wrapper
│   │   └── export_model_for_selfplay.sh
│   ├── tests/                     # Python unit tests
│   │   └── test_training_data_generator.py
│   ├── train.py                   # Main PyTorch training entry point
│   ├── shuffle.py                 # Multi-threaded selfplay data shuffling script
│   └── export_model_pytorch.py    # Checkpoint exporter to KataGo .bin.gz format
├── docs/                          # Documentation (Analysis Engine, GTP extensions, GraphSearch, etc.)
├── .clang-format                  # C++ code formatting rules (Chromium base, column limit 120, 2-space indent)
└── pytest.ini                     # Pytest configuration rooted at python/tests
```

---

## 4. Build and Compilation Guide

### CMake Flags
The primary CMake file is `cpp/CMakeLists.txt`. Key build options:
- `-DDOTS_GAME=1` (Default: `1`): Configures compilation for Dots with default max board length `COMPILE_MAX_BOARD_LEN_X=39` and `COMPILE_MAX_BOARD_LEN_Y=32`.
- `-DUSE_BACKEND=<BACKEND>`:
  - `TENSORRT`: Recommended for NVIDIA GPUs (supports TensorRT 10 and 11, builds via ONNX parser).
  - `CUDA`: Alternative NVIDIA GPU backend with cuDNN.
  - `OPENCL`: General GPU backend (NVIDIA, AMD, Intel).
  - `EIGEN`: CPU-only backend. Add `-DUSE_AVX2=1` on modern x86_64 processors for significant speedup.
  - `METAL`: Apple Silicon MPSGraph + CoreML backend.
- `-DCMAKE_BUILD_TYPE=Release`: Use `Release` for optimal performance.

### Windows (MSVC) Build
Prerequisites: Visual Studio 2019/2022 (Desktop development with C++), CMake >= 3.18.2, vcpkg or prebuilt dependencies (`zlib`, `libzip`, `protobuf`, `TensorRT` if using TensorRT).

```powershell
# Example: Building TensorRT backend with existing build tree in cpp/builds/trt-release
cd cpp/builds/trt-release
cmake --build . --config Release -j 4

# Or configuring a fresh build with vcpkg:
cd cpp
mkdir build; cd build
cmake .. -A x64 `
  -DUSE_BACKEND=OPENCL `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build . --config Release -j 4
```

> **Important (Windows DLLs)**: When running `katago.exe`, ensure the required runtime DLLs (`z.dll`, `zip.dll`, `libprotobuf.dll`, `abseil_dll.dll`, `bz2.dll`, `nvinfer.dll` etc.) are placed in the same directory as the executable.

### Linux Build
```bash
cd cpp
cmake . -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## 5. Running Tests

### 1. C++ Built-in Test Suite
Run the built-in unit tests and stress tests directly via `katago runtests`:
```powershell
# From cpp/ directory (or pointing to compiled binary):
.\katago.exe runtests
```
This comprehensive suite runs:
- Inline and file config parsing tests
- Mathematical, hashing, and RNG tests
- Dots field logic, cycle tracing, base boundaries, and empty territory tests
- Dots grounding checks and board history grounding tests
- Start position generator and recognizer tests
- Dots symmetry and komi randomization tests
- Dots NN input preparation tests
- High-volume random Dots games stress tests (100k+ simulated games)

Additional C++ test commands:
- `.\katago.exe runoutputtests` — Tests model loading, SGF handling, time controls, and output formatting.
- `.\katago.exe version` — Prints engine version, git commit, build type, backend, and board dimensions.

### 2. C++ Google Test Suite
When compiled with `katago_tests`:
```powershell
Release/katago_tests.exe
```
Discovers and tests `tests/testdotsladders.cpp`, `tests/testdotsutils.cpp`, and `tests/testdotsstress.cpp`.

### 3. Python Unit Tests
The repository uses `pytest` configured in `pytest.ini`:
```powershell
# From repo root:
pytest
```
Runs unit tests for the data generator and shuffler in `python/tests/test_training_data_generator.py`.

---

## 6. Running KataGoDots Engine

### GTP Engine (for GUIs & Controllers)
Dots games can be played over GTP using `cpp/configs/gtp_dots.cfg`:
```powershell
.\katago.exe gtp -config cpp/configs/gtp_dots.cfg -model <MODEL_PATH>.bin.gz
```
- Supports Bronstein delay (`time_settings <main> <per_move>`, `time_left <color> <main_left> <delay_left>`).
- Supports `kata-time_settings bronstein <main> <delay>`.
- Allows multiple move generation via `genmove`.

### JSON Analysis Engine (for Backends & Tools)
High-throughput parallel board analysis using `cpp/configs/analysis_dots.cfg`:
```powershell
.\katago.exe analysis -config cpp/configs/analysis_dots.cfg -model <MODEL_PATH>.bin.gz
```
- Query fields:
  - `"dots": true` — Declares the query as a Dots game.
  - `"playerToMove": "b"` or `"w"` — Specifies whose turn to evaluate.
  - `"rules": "bbs"` or `"notago"`.
  - `"initialStones"`: Placed dots for the initial position.
- Response fields:
  - `"chosenMove"`: The move chosen by the engine according to chosenMoveTemperature.
  - `"resignReasonable"`: Boolean indicating whether resigning is objectively justified based on grounding/captures.

### Performance Benchmark & Thread Tuning
```powershell
.\katago.exe benchmark -config cpp/configs/gtp_dots.cfg -model <MODEL_PATH>.bin.gz
```
Recommends optimal `numSearchThreads` for your hardware.

---

## 7. Training Pipeline (Self-play Loop)

For selfplay training on a single machine, use `python/selfplay/synchronous_loop.sh`:
```bash
./python/selfplay/synchronous_loop.sh <RUN_NAME> <BASE_DIR> <TRAINING_NAME> <MODEL_KIND> <USE_GATING>
```
Example:
```bash
./python/selfplay/synchronous_loop.sh dotsrun /d/Dots/training run1 b6c96 0
```
The loop carries out five stages sequentially:
1. **Gatekeeper** (optional, C++): Matches candidate model against the current best.
2. **Selfplay** (C++): Plays games using current model and dumps training data.
3. **Shuffle** (Python): Multi-threaded shuffle of raw selfplay data into training `.npz` buckets.
4. **Train** (PyTorch): Trains neural net on shuffled data with weight averaging (SWA).
5. **Export** (Python): Converts PyTorch checkpoint to optimized `.bin.gz` for C++.

---

## 8. Development Guidelines & Gotchas for AI Agents

1. **Always Respect Rectangular Coordinates (`x_size` vs `y_size`)**:
   - In Go, boards are almost always square. In Dots, boards are usually rectangular (e.g. 39x32).
   - In C++, board location indexing uses stride `(x_size + 1)`:
     `Location::getLoc(x, y, x_size) = (x + 1) + (y + 1) * (x_size + 1)`.
   - Always check bounds separately for X (`0 <= x < x_size`) and Y (`0 <= y < y_size`). Reject out-of-range coordinates.
   - Do not assume symmetry across diagonal unless `x_size == y_size`.

2. **Dots Rules & Terminal Conditions**:
   - In Dots, consecutive passes do **not** simply end the game with territory scoring like Go.
   - A pass in Dots is interpreted as a grounding attempt (`PASS_LOC`). Grounding logic is handled in `dotsboardhistory.cpp` via `whiteScoreIfGroundingAlive`.
   - Never replace `Board` parameters by mutable reference if `hist.initialBoard` could be inadvertently mutated.

3. **Neural Net Features and Heads**:
   - Spatial input features (`cpp/neuralnet/nninputsdots.cpp`): 22 channels defined by `DotsSpatialFeature`. Note: Ladder features (`LadderCaptured_14` through `LadderWorkingMoves_17`) are currently temporarily disabled in feature filling.
   - Global input features: 19 channels defined by `DotsGlobalFeature`.
   - Value / auxiliary heads: KataGo's original Go-specific seki and score distribution heads were removed in commit `885e426f`. Do not reintroduce them without updating model definitions in both C++ and Python (`model_pytorch.py`, `load_model.py`, `train.py`).
   - Ownership prediction: Masked in `nneval.cpp` and `model_pytorch.py` to only placed dots.

4. **Code Formatting & Cleanliness**:
   - **C++**: Follow the rules defined in `.clang-format` (Chromium base, 2-space indentation, 120 characters column limit). Keep headers alphabetically sorted.
   - **Python**: Follow PEP 8 (4 spaces indentation). Keep tests under `python/tests/` and verify with `pytest`.
   - **Git Branching**: The default working branch in this repository is `master-mine`. Ensure changes do not break CI checks defined in `.github/workflows/build.yml`.
