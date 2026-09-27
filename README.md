# KataGoDots

[![Build](https://github.com/KvanTTT/KataGoDots/actions/workflows/build.yml/badge.svg)](https://github.com/KvanTTT/KataGoDots/actions/workflows/build.yml)
[![ONNX Backend CI](https://github.com/KvanTTT/KataGoDots/actions/workflows/onnx-backend.yml/badge.svg)](https://github.com/KvanTTT/KataGoDots/actions/workflows/onnx-backend.yml)

KataGoDots is a fork and adaptation of [KataGo](https://github.com/lightvector/KataGo) (v1.17+) engineered specifically for the game of **Dots** (Russian: *Точки* / *Игра в точки*), while maintaining underlying support for Go.

- **Primary Maintainers**: Ivan Kochurkin ([@KvanTTT](https://github.com/KvanTTT)), Nikita Shokarev ([@nikitos4683](https://github.com/nikitos4683))
- **Original KataGo Author**: David J. Wu ([@lightvector](https://github.com/lightvector))
- **Repository**: [https://github.com/KvanTTT/KataGoDots](https://github.com/KvanTTT/KataGoDots)

---

## Table of Contents

- [Overview & Game Mechanics](#overview--game-mechanics)
  - [Dots vs. Go](#dots-vs-go)
  - [Rulesets and Presets](#rulesets-and-presets)
- [Running KataGoDots](#running-katagodots)
  - [GTP Engine](#gtp-engine)
  - [JSON Analysis Engine](#json-analysis-engine)
  - [Performance Benchmarking](#performance-benchmarking)
  - [Running Tests](#running-tests)
- [Backends: OpenCL vs. CUDA vs. TensorRT vs. ROCm vs. Eigen vs. ONNX vs. Metal](#backends)
- [Building & Compilation](#building--compilation)
- [Features for Developers & GUIs](#features-for-developers--guis)
  - [GTP Extensions](#gtp-extensions)
  - [JSON Analysis Protocol](#json-analysis-protocol)
- [Selfplay Training Pipeline](#selfplay-training-pipeline)
- [Source Code Overview](#source-code-overview)
- [License](#license)

---

## Overview & Game Mechanics

KataGoDots adapts KataGo's MCTS search, territory evaluation, feature extraction, and neural network training pipeline to the rules and dynamics of the game of Dots.

### Dots vs. Go

| Feature | Go | Dots (*Точки*) |
| :--- | :--- | :--- |
| **Grid / Board Size** | Typically 19x19, 13x13, 9x9 (square) | Rectangular standard default **39x32** (`COMPILE_MAX_BOARD_LEN_X=39`, `COMPILE_MAX_BOARD_LEN_Y=32`), configurable (e.g. 20x20) |
| **Move Placement** | Stones on intersections | Dots on intersections |
| **Captures / Territory** | Surrounding stones removes them from the board | Surrounding opponent dots forms an enclosure (**base / база**). Enclosed dots remain on the board as captured points; bases cannot be recaptured |
| **Pass / Grounding** | Pass move; two consecutive passes end the game | Pass (`PASS_LOC`) represents **grounding (заземление)** — connecting a base to the border or grounded area. A game ends if a player grounds alive or the position is terminal |
| **Start Positions** | Usually empty board (or handicap stones) | Presets: `EMPTY` (0), `SINGLE` (1), `CROSS` (2, 1 скрест), `CROSS_2` (3), `CROSS_4` (4, 4 скреста), with optional random perturbation |
| **Time Controls** | Byo-yomi, Canadian, Fischer | **Bronstein delay** (`time_settings <main> <per_move>`): each move uses per-move delay first; unused delay is not banked |
| **Expected Game Length** | ~250 moves on 19x19 (~0.69 moves/point) | ~400 moves on 39x32 (~0.32 moves/point). Move temperature decay is normalized against Dots expected length |
| **Neural Net Inputs** | 22 spatial features, 19 global features | 22 spatial features (including Dots tactical ladder planes 14–17) and 19 global features |

### Rulesets and Presets

KataGoDots includes built-in presets for common Dots rulesets:
* `bbs` (or `russian`): Standard single cross (`CROSS`) start position, matching classic Russian server rules (e.g., playdots.ru).
* `notago`: 4 random crosses (`CROSS_4`) start position, matching Notago server rules.

---

## Running KataGoDots

KataGoDots is a command-line engine that communicates via GTP (Go Text Protocol with Dots extensions) or a JSON-based parallel analysis protocol.

### GTP Engine

Dots games can be played over GTP using the provided Dots configuration:
```powershell
./katago gtp -config cpp/configs/gtp_dots.cfg -model <MODEL_PATH>.bin.gz
```

Notable features:
* Supports Bronstein delay (`time_settings <main> <per_move>`, `time_left [color] <main_left> <delay_left>`).
* Custom GTP commands: `get_boardsize`, `get_moves`, `get_position`, multi-move `play`, `undo [count]`, multi-move `genmove [color] [moves_count]`, and `info`.
* Rules configuration: `kata-set-rules bbs` or `kata-set-rules notago`.

### JSON Analysis Engine

High-throughput parallel board analysis using `cpp/configs/analysis_dots.cfg`:
```powershell
./katago analysis -config cpp/configs/analysis_dots.cfg -model <MODEL_PATH>.bin.gz
```

Queries are JSON lines on stdin, and results are JSON lines on stdout:
```json
{"id":"pos-1","boardXSize":39,"boardYSize":32,"moves":[],"rules":"bbs","dots":true}
```

Dots-specific fields:
* `"dots": true`: Overrides the game selection (already defaulted in `analysis_dots.cfg`).
* `"playerToMove": "b"` or `"w"`: Specifies whose turn to evaluate (in Dots, either player may move).
* `"initialStones"`: Pre-placed dots for the initial position.
* `"chosenMove"`: The move chosen by the engine according to `chosenMoveTemperature`.
* `"resignReasonable"`: Boolean indicating whether resigning is objectively justified based on grounding and capture state.

### Performance Benchmarking

Benchmark your hardware and find the optimal `numSearchThreads`:
```powershell
./katago benchmark -config cpp/configs/gtp_dots.cfg -model <MODEL_PATH>.bin.gz
```

### Running Tests

Run the built-in unit tests and stress tests directly via `katago runtests`:
```powershell
cd cpp
./katago runtests
```

This suite verifies:
* Inline and file config parsing.
* Mathematical, hashing, and RNG logic.
* Dots field logic, cycle tracing, base boundaries, and empty territory.
* Dots tactical ladder solver (`testdotsladders.cpp`).
* Dots grounding checks and board history grounding tests.
* Start position generators and recognizers (`testdotsstartposes.cpp`).
* Dots symmetry and komi randomization.
* Dots neural net input preparation (`nninputsdots.cpp`).
* High-volume random Dots games stress tests (100k+ simulated games).

---

## Backends

KataGoDots supports multiple computation backends:
* **OpenCL**: General GPU backend compatible with NVIDIA, AMD, and Intel GPUs. Requires initial auto-tuning on first launch.
* **CUDA**: Native NVIDIA GPU backend with cuDNN.
* **TensorRT**: High-performance NVIDIA GPU backend using TensorRT (supports TensorRT 10 and 8.5+ via ONNX parser).
* **ROCm**: AMD GPU backend supporting CDNA and RDNA GPUs on Linux and Windows.
* **Eigen**: CPU-only backend. Supports `-DUSE_AVX2=1` for significant speedups on modern x86_64 processors.
* **ONNX Runtime**: Cross-platform backend using ONNX Runtime execution providers (e.g. OpenVINO, DirectML).
* **Metal**: Apple Silicon backend for macOS.

---

## Building & Compilation

KataGoDots requires a C++17 compliant compiler and CMake >= 3.18.2.

Key CMake options:
* `-DDOTS_GAME=1` (Default: `1`): Configures compilation for Dots with default board size `COMPILE_MAX_BOARD_LEN_X=39` and `COMPILE_MAX_BOARD_LEN_Y=32`.
* `-DUSE_BACKEND=<BACKEND>`: `OPENCL`, `CUDA`, `TENSORRT`, `ROCM`, `EIGEN`, `ONNX`, or `METAL`.
* `-DUSE_AVX2=1`: Enables AVX2 and FMA optimizations when using the `EIGEN` CPU backend.

### Quick Build Example (Linux / OpenCL)
```bash
git clone https://github.com/KvanTTT/KataGoDots.git
cd KataGoDots/cpp
cmake . -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Quick Build Example (Windows / MSVC with vcpkg)
```powershell
cmake -S cpp -B cpp/builds/msvc-opencl -G "Visual Studio 17 2022" -A x64 `
  -DUSE_BACKEND=OPENCL `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build cpp/builds/msvc-opencl --config Release -j 4
```

See [Compiling.md](Compiling.md) for full instructions covering Windows, Linux, macOS, and each GPU backend.

---

## Features for Developers & GUIs

### GTP Extensions

KataGoDots provides several extensions to standard GTP:
* `get_boardsize`: Returns current board dimensions `X` or `X:Y`.
* `get_moves`: Returns the sequence of moves played so far.
* `get_position`: Returns the start position moves.
* `play COLOR VERTEX [COLOR VERTEX ...]`: Plays one or more moves in sequence, rolling back atomically on error.
* `undo [COUNT]`: Undoes `COUNT` moves (defaults to 1).
* `genmove [COLOR] [MOVES_COUNT]`: Generates up to `MOVES_COUNT` moves in sequence.
* `info`: Returns engine and build metadata in CSV format (`app_name,app_version,git_rev,compile_datetime,backend,max_len_x,max_len_y,build_type`).
* `time_settings MAINTIME PERMOVETIME`: Sets Bronstein delay time control for Dots.
* `time_left [COLOR] TIME [PERMOVETIMELEFT]`: Reports remaining main time and move delay.
* `kata-time_settings bronstein MAINTIME DELAY` / `kata-time_settings default`: Exposes Bronstein delay configuration.
* `kata-get-rules` / `kata-set-rules [bbs|notago]`: Inspects and sets game rules.

For detailed command specifications, see [docs/GTP_Extensions.md](docs/GTP_Extensions.md).

### JSON Analysis Protocol

The parallel analysis engine accepts JSON queries on stdin and outputs JSON responses on stdout. See [docs/Analysis_Engine.md](docs/Analysis_Engine.md) for full protocol documentation.

---

## Selfplay Training Pipeline

KataGoDots includes a complete pipeline for training neural networks through selfplay:
1. **Gatekeeper** (C++): Matches candidate models against the current best.
2. **Selfplay** (C++): Simulates selfplay games and writes training data using `cpp/configs/training/selfplay1_dots.cfg`.
3. **Shuffle** (Python): Multi-threaded shuffling of selfplay `.npz` records.
4. **Train** (PyTorch): Trains the model using rectangular coordinate handling (`pos_len_x`, `pos_len_y`) and rectangle-preserving symmetries.
5. **Export** (Python): Converts PyTorch checkpoints to `.bin.gz` models for the C++ engine via `python/export_model_pytorch.py`.

A single-machine training loop is provided in `python/selfplay/synchronous_loop.sh`:
```bash
./python/selfplay/synchronous_loop.sh <NAMEPREFIX> <BASEDIR> <TRAININGNAME> <MODELKIND> <USEGATING>
```

See [SelfplayTraining.md](SelfplayTraining.md) and [python/README.md](python/README.md) for detailed training setup and parameter tuning.

---

## Source Code Overview

* `cpp/`: C++ engine, MCTS search, Dots cycle and territory tracing, tactical ladder solver, and inference backends. See [cpp/README.md](cpp/README.md).
* `python/`: Neural network definitions, training pipeline, rectangular data processing, and export scripts. See [python/README.md](python/README.md).
* `docs/`: Technical specifications for the analysis engine, GTP extensions, network architectures, and graph search.

---

## License

All original modifications for KataGoDots are released under the same open-source license as KataGo ([LICENSE](LICENSE)). See [CONTRIBUTORS](CONTRIBUTORS) for list of contributors.
