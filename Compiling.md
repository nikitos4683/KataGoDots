# Compiling KataGoDots

KataGoDots is written in C++17. It compiles on Linux or macOS via g++ / clang++ supporting at least C++17, or on Windows via MSVC (Visual Studio 2019 / 2022) or MinGW.

The repository is hosted at: [https://github.com/KvanTTT/KataGoDots](https://github.com/KvanTTT/KataGoDots)

---

## CMake Configuration & Flags

KataGoDots uses CMake (minimum version 3.18.2). All build configuration takes place in `cpp/`.

Key CMake flags:
* `-DDOTS_GAME=1` (Default: `1`): Configures compilation for Dots with default max board length `COMPILE_MAX_BOARD_LEN_X=39` and `COMPILE_MAX_BOARD_LEN_Y=32`. Custom board sizes can be specified via `-DCOMPILE_MAX_BOARD_LEN_X=...` and `-DCOMPILE_MAX_BOARD_LEN_Y=...`.
* `-DUSE_BACKEND=<BACKEND>`:
  * `OPENCL`: General GPU backend (NVIDIA, AMD, Intel). Easiest to set up; auto-tunes on first run.
  * `CUDA`: NVIDIA GPU backend using CUDA and cuDNN.
  * `TENSORRT`: High-performance NVIDIA GPU backend using TensorRT (requires TensorRT 10 or 8.5+).
  * `ROCM`: AMD GPU backend using ROCm / MIOpen.
  * `EIGEN`: CPU-only backend. Add `-DUSE_AVX2=1` on modern x86_64 processors for significant performance boost.
  * `ONNX`: Cross-platform backend via ONNX Runtime execution providers.
  * `METAL`: Apple Silicon backend using MPSGraph and CoreML.
* `-DCMAKE_BUILD_TYPE=Release`: Recommended for optimal search speed and selfplay.

---

## Linux

### Quick Build (OpenCL)
```bash
git clone https://github.com/KvanTTT/KataGoDots.git
cd KataGoDots/cpp
cmake . -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Quick Build (CPU / Eigen with AVX2)
```bash
git clone https://github.com/KvanTTT/KataGoDots.git
cd KataGoDots/cpp
cmake . -DUSE_BACKEND=EIGEN -DUSE_AVX2=1 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Requirements (Linux)
* **CMake** >= 3.18.2 (`sudo apt install cmake`).
* **Compiler**: GCC >= 9 or Clang >= 10 supporting C++17.
* **Libraries**:
  * `zlib1g-dev`, `libzip-dev` (for model loading, SGF handling, and selfplay data writing).
  * `libgoogle-perftools-dev` (recommended for TCMalloc; pass `-DUSE_TCMALLOC=1`).
* **Backend dependencies**:
  * **OpenCL**: OpenCL 1.2+ headers and drivers (`ocl-icd-opencl-dev`, `opencl-headers`).
  * **CUDA**: CUDA Toolkit 11+ and compatible cuDNN (`nvidia-cuda-toolkit`, `libcudnn8-dev`).
  * **TensorRT**: CUDA Toolkit and TensorRT 10 or 8.5+ (`libnvinfer-dev`, `libnvonnxparser-dev`).
  * **ROCm**: ROCm 6.4+ developer stack (`rocm-dev`, `miopen-hip-dev`, `hipblas-dev`, `rocblas-dev`).
  * **Eigen**: `libeigen3-dev`.

### ROCm Backend (Linux) - Additional Notes
* Install full ROCm developer packages: `sudo apt install rocm-dev miopen-hip-dev hipblas-dev rocblas-dev`.
* Build:
  ```bash
  cd KataGoDots/cpp
  mkdir build && cd build
  cmake .. -DUSE_BACKEND=ROCM -DCMAKE_BUILD_TYPE=Release
  make -j$(nproc)
  ```
* The build auto-detects `/opt/rocm/core-<version>/` or `/opt/rocm/`. Override with `-DCMAKE_PREFIX_PATH=...` if needed.
* Target architecture: Default targets Vega 20, CDNA, and RDNA. Specify `-DCMAKE_HIP_ARCHITECTURES=gfx1100` to target only your specific GPU.

---

## Windows

### Prerequisites
* **Visual Studio 2019 or 2022** (Desktop development with C++).
* **CMake** >= 3.18.2.
* **Dependencies via [vcpkg](https://github.com/microsoft/vcpkg)** (recommended):
  ```powershell
  git clone https://github.com/microsoft/vcpkg.git
  cd vcpkg
  .\bootstrap-vcpkg.bat
  .\vcpkg.exe install zlib:x64-windows libzip:x64-windows protobuf:x64-windows
  ```

### Build with Visual Studio & vcpkg

From the repository root:
```powershell
# Configure with OpenCL backend:
cmake -S cpp -B cpp/builds/msvc-opencl -G "Visual Studio 17 2022" -A x64 `
  -DUSE_BACKEND=OPENCL `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

# Build Release binary:
cmake --build cpp/builds/msvc-opencl --config Release -j 4
```

For the CPU backend with AVX2:
```powershell
cmake -S cpp -B cpp/builds/msvc-eigen -G "Visual Studio 17 2022" -A x64 `
  -DUSE_BACKEND=EIGEN `
  -DUSE_AVX2=1 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build cpp/builds/msvc-eigen --config Release -j 4
```

### Runtime DLLs
When running `katago.exe`, ensure the required runtime DLLs (`z.dll`, `zip.dll`, `libprotobuf.dll`, etc.) are in the same folder as the executable or on `PATH`. With vcpkg, they can be found in `vcpkg\installed\x64-windows\bin\`.

### MinGW (Alternative Windows Toolchain)
Using [MSYS2](https://www.msys2.org/):
```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-libzip mingw-w64-x86_64-zlib
```
Note: CUDA and TensorRT backends are not supported under MinGW. Use MSVC for NVIDIA backends.

---

## macOS

### Metal Backend (Recommended for Apple Silicon)
```bash
git clone https://github.com/KvanTTT/KataGoDots.git
cd KataGoDots/cpp
cmake -G Ninja -DUSE_BACKEND=METAL -DCMAKE_BUILD_TYPE=Release
ninja
```

Prerequisites via Homebrew:
```bash
brew install cmake ninja protobuf abseil libzip
```

---

## ONNX Runtime Backend (Optional)

The `ONNX` backend runs models through [ONNX Runtime](https://onnxruntime.ai/) execution providers (e.g. OpenVINO on Intel GPUs/NPUs, DirectML on DirectX 12 GPUs).

Compile:
```bash
cmake -S cpp -B cpp/build -DUSE_BACKEND=ONNX -DONNXRUNTIME_ROOT=<path-to-onnxruntime>
cmake --build cpp/build -j
```

See [docs/ONNX_Model_Files.md](docs/ONNX_Model_Files.md) for details on working with `.onnx` models and `katago dumponnx`.

---

## Verification After Building

Once built, verify your executable:

```powershell
# 1. Print version and compile configuration:
./katago version

# 2. Run built-in test suite:
./katago runtests

# 3. Benchmark search speed:
./katago benchmark -config configs/gtp_dots.cfg -model <MODEL>.bin.gz
```
