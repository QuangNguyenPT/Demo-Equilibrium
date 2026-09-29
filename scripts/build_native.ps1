# Build the HRM-X native backend (C ABI, loaded by hrm_x/native via ctypes).
#   .\scripts\build_native.ps1            # release build -> hrm_x\native\bin\hrmx_native.dll
#   .\scripts\build_native.ps1 -Debug     # -O0 -g
# Follows the spec v0.4 §23 native-backend convention (ADR-008): Python is the blueprint and
# reference semantics, C++ executes the profiled hot paths behind a pure C ABI. CPU only:
# the hot paths are pointer-chasing symbolic code, not matrix math (spec v0.4 §23).
param([switch]$Debug)

$Root = Split-Path $PSScriptRoot -Parent
$Native = Join-Path $Root "hrm_x\native"
$Bin = Join-Path $Native "bin"
New-Item -ItemType Directory -Force -Path $Bin | Out-Null

$Clang = (Get-Command clang++ -ErrorAction SilentlyContinue).Source
if (-not $Clang) { $Clang = "C:\Program Files\LLVM\bin\clang++.exe" }
if (-not (Test-Path -LiteralPath $Clang)) {
    Write-Host "[NATIVE] clang++ not found; the Python reference backend stays in use." -ForegroundColor Yellow
    exit 1
}
$Opt = if ($Debug) { @("-O0", "-g") } else { @("-O2", "-DNDEBUG") }
$Sources = Get-ChildItem (Join-Path $Native "src") -Filter *.cpp | ForEach-Object { $_.FullName }
$Out = Join-Path $Bin "hrmx_native.dll"

& $Clang -std=c++17 @Opt -Wall -Wextra -shared -I (Join-Path $Native "include") @Sources -o $Out
if ($LASTEXITCODE -ne 0) {
    Write-Host "[NATIVE] build failed (exit $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[NATIVE] built $Out" -ForegroundColor Green
