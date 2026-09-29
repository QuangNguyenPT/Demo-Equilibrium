---
title: Build & Toolchain Guide
type: guide
tags:
  - guide
  - build
  - native
  - clang
  - ctypes
status: active
created: 2026-09-26
updated: 2026-09-28
relates_to:
  - "[[ADR-001 Hybrid C++ Engine with Python Blueprint]]"
  - "[[HRM-X Native Backend]]"
---

# 🛠️ Build & Toolchain Guide

> [!NOTE] Corrected 2026-09-28
> The build instructions below used to describe the ADR-001 engine: a CMake project at the repo
> root, `src/core`, `src/kernels`, `src/runtime`, `apps/cli/main.cpp` and a `hrm_cli.exe`. **None
> of that exists** — it was never integrated and lives in `legacy/cpp_engine/`. The commands in
> the Python and C++ sections below are the only ones that work today. There is no CLI binary and
> no blueprint layer to test.

---

## 🐍 Python

```powershell
# From the repo root
python -m pytest                      # the whole suite: 436 passed, 1 skipped
python -m pytest tests/test_x_native.py -v   # native-vs-Python parity
```

Dependencies are declared in `pyproject.toml`. There is no separate requirements file and no
`python/` package to install — `python/` at the repo root is **empty**, and so is `configs/`.
The model packages are imported from the repo root: `hrm/`, `hrm_eqb/`, `hrm_x/`.

---

## ⚙️ C++ — the HRM-X native backend

There is exactly one C++ target in this project: `hrm_x/native/`, the measured hot paths
(e-graph, grid) behind the `hrmx.h` C ABI. It is a two-file library with one build script.

```powershell
# Release: clang++ -O2 -DNDEBUG -std=c++17 -shared  ->  hrm_x\native\bin\hrmx_native.dll
.\scripts\build_native.ps1

# Debug: -O0 -g
.\scripts\build_native.ps1 -Debug
```

The script is the sanctioned path and it prefers `clang++` on `PATH`, falling back to
`C:\Program Files\LLVM\bin\clang++.exe`. The command it runs, in effect:

```powershell
& clang++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -shared `
    -I hrm_x\native\include `
    hrm_x\native\src\egraph.cpp hrm_x\native\src\grid.cpp `
    -o hrm_x\native\bin\hrmx_native.dll
```

- **Sources**: `hrm_x/native/src/*.cpp` (currently `egraph.cpp`, `grid.cpp`).
- **Include**: `hrm_x/native/include/hrmx.h` only. No third-party dependencies.
- **Output**: `hrm_x/native/bin/hrmx_native.dll` (git-ignored; the Python reference is used
  automatically when it is missing).
- **CPU only.** No CUDA, no BLAS, no OpenMP. The hot paths are pointer-chasing symbolic code,
  not matrix math (spec v0.4 §23).
- **No CMake, no MSVC project, no `hrm_cli.exe`.** The old `CMakeLists.txt` is at
  `legacy/cpp_engine/CMakeLists.txt` and builds nothing used by the model.

### Using the native backend

```python
import hrm_x.native as native

native.available()    # True once the DLL is built
native.enabled()      # False if HRMX_BACKEND=python
```

```powershell
# Force the Python reference (e.g. to debug a discrepancy)
$env:HRMX_BACKEND = "python"
```

---

## 📜 Historical — the ADR-001 build, never integrated

Kept for the record. These commands will **not** work from the repo root; they only work from
`legacy/cpp_engine/`, whose output nothing consumes.

<details>
<summary><code>legacy/cpp_engine</code> — clang one-liner (as written in 2026-09-26)</summary>

```powershell
# Create output directory
New-Item -ItemType Directory -Force -Path "bin"

# Compile shared library DLL
& "C:\Program Files\LLVM\bin\clang++.exe" -std=c++17 -O3 -I include `
    src/core/*.cpp src/kernels/*.cpp src/runtime/*.cpp src/c_api/*.cpp `
    -shared -o bin/hrm_engine.dll

# Compile standalone CLI executable
& "C:\Program Files\LLVM\bin\clang++.exe" -std=c++17 -O3 -I include `
    apps/cli/main.cpp src/core/*.cpp src/kernels/*.cpp src/runtime/*.cpp `
    -o bin/hrm_cli.exe

# Run the native C++ executable
.\bin\hrm_cli.exe
```

</details>

<details>
<summary>CMake (as written in 2026-09-26)</summary>

```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

# Configure
& $cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Build
& $cmake --build build --config Release
```

Linux / macOS, as written in 2026-09-26:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -- -j$(nproc)
```

</details>

<details>
<summary>Blueprint layer tests (as written in 2026-09-26)</summary>

`python -m pytest tests/test_blueprint.py -v` and `python scripts/run_demo.py` targeted
`python/hrm_blueprint/`. Both files live at `legacy/tests/test_blueprint.py` and
`legacy/scripts/run_demo.py` now, and neither is collected by the current suite.

</details>

---

## 🔗 Related Notes
- [[HRM-X Native Backend]] — what `hrm_x/native/` contains.
- [[C-ABI & Interop Specification]] — the `hrmx.h` surface the DLL exports.
- [[Development Workflow & Conventions]] — logging standards and code style.
- [[ADR-001 Hybrid C++ Engine with Python Blueprint]] — the design this guide used to serve.
