---
title: Build & Toolchain Guide
type: guide
tags:
  - guide
  - build
  - cmake
  - clang
  - msvc
status: active
created: 2026-09-26
---

# 🛠️ Build & Toolchain Guide

This guide details how to compile the native C++ engine, build the C-ABI shared library, and execute tests across Windows and Linux platforms.

---

## 🪟 Windows Build Environments

### Option 1: Direct Clang++ One-Liner (Fastest & Simplest)
Clang++ is available at `C:\Program Files\LLVM\bin\clang++.exe`. You can compile the shared engine DLL directly without configuring a build generator:

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

---

### Option 2: CMake + Visual Studio / Ninja

If using CMake bundled with Visual Studio:
```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

# Configure
& $cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Build
& $cmake --build build --config Release
```

---

## 🐧 Linux / macOS Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -- -j$(nproc)
```

---

## 🧪 Testing the Python Blueprint Layer

Run the automated pytest suite:
```powershell
python -m pytest tests/test_blueprint.py -v
```

Run the end-to-end execution demonstration:
```powershell
python scripts/run_demo.py
```

---

## 🔗 Related Notes
- [[Development Workflow & Conventions]] — Logging standards and code style.
- [[Engine Architecture & Memory Model]] — Memory allocator and thread pool.
- [[C-ABI & Interop Specification]] — Interoperability details.
