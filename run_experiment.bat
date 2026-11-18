@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo EXPERIMENT 5: NEURAL NETWORK LAYERS IN CUDA
echo Automated Execution and Benchmarking Suite
echo ======================================================================

set "BASE_DIR=%~dp0"
cd /d "%BASE_DIR%"

if not exist "results\execution_logs" mkdir "results\execution_logs"
if not exist "results\graphs" mkdir "results\graphs"

echo [1/5] Checking System Environment and Compilers...

set "HAS_NVCC=0"
where nvcc >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] CUDA Compiler nvcc detected in PATH.
    set "HAS_NVCC=1"
) else (
    echo [NOTE] nvcc not found in default PATH.
)

set "HAS_CL=0"
where cl >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] MSVC C++ Compiler cl.exe detected.
    set "HAS_CL=1"
) else (
    echo [INFO] Searching for Visual Studio Build Tools...
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
        set "HAS_CL=1"
        echo [OK] Initialized Visual Studio 2026 x64 Developer Tools.
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
        set "HAS_CL=1"
        echo [OK] Initialized Visual Studio 2022 x64 Developer Tools.
    )
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] Python runtime detected.
) else (
    echo [WARNING] Python not detected in PATH.
)

echo.
echo [2/5] Compiling Experiment Implementations...

if "!HAS_NVCC!"=="1" (
    echo Compiling native CUDA source: nvcc -O2 src/neural_network.cu -o neural_network.exe
    nvcc -O2 src/neural_network.cu -o neural_network.exe
    if %errorlevel% neq 0 (
        echo [ERROR] CUDA compilation failed.
        exit /b 1
    )
    echo [OK] Successfully built neural_network.exe.
    set "RUN_TARGET=neural_network.exe"
) else (
    echo Compiling C++ validation suite: cl /O2 /EHsc /W4 src/cpu_validation.cpp /Fe:cpu_validation.exe
    cl /O2 /EHsc /W4 src/cpu_validation.cpp /Fe:cpu_validation.exe
    if %errorlevel% neq 0 (
        echo [ERROR] C++ compilation failed.
        exit /b 1
    )
    echo [OK] Successfully built cpu_validation.exe.
    set "RUN_TARGET=cpu_validation.exe"
)

echo.
echo [3/5] Executing Validation and Benchmarks...
echo Running !RUN_TARGET! ...
"!RUN_TARGET!" > "results\execution_logs\latest_run.log"
type "results\execution_logs\latest_run.log"

echo.
echo [4/5] Executing Python Validation and Performance Benchmarking...
python scripts\validate_results.py
python scripts\run_benchmarks.py

echo.
echo [5/5] Generating Publication Graphs and Visualizations...
python scripts\generate_graphs.py

echo.
echo ======================================================================
echo Experiment 5 Execution and Verification Completed Successfully.
echo Results stored in results/ and graphs in results/graphs/.
echo ======================================================================

rem Clean up intermediate build objects
if exist "cpu_validation.obj" del /f /q "cpu_validation.obj"
if exist "cpu_validation.exe" del /f /q "cpu_validation.exe"

exit /b 0
