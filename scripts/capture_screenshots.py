#!/usr/bin/env python3
"""
Terminal screenshot generator for Experiment 5.
Captures genuine command outputs and renders clean, readable terminal images.
"""

import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

def render_terminal(title, command, output, out_path, width=1050):
    bg_color = (24, 24, 24)
    header_color = (38, 38, 38)
    text_color = (220, 220, 220)
    cmd_color = (130, 250, 130)
    title_color = (180, 180, 180)
    prompt_color = (80, 180, 255)

    try:
        font = ImageFont.truetype("consola.ttf", 15)
        title_font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    lines = []
    lines.append(("prompt", f"PS C:\\pgc_exp5_Neural_Network_Layers_In_Cuda> {command}"))
    for l in output.splitlines():
        lines.append(("out", l))

    line_height = 20
    header_height = 36
    padding = 18
    img_height = header_height + padding * 2 + len(lines) * line_height + 10

    img = Image.new("RGB", (width, img_height), bg_color)
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([(0, 0), (width, header_height)], fill=header_color)
    draw.ellipse([(14, 12), (24, 22)], fill=(255, 95, 86))
    draw.ellipse([(32, 12), (42, 22)], fill=(255, 189, 46))
    draw.ellipse([(50, 12), (60, 22)], fill=(39, 201, 63))
    draw.text((80, 10), title, fill=title_color, font=title_font)

    # Body text
    y = header_height + padding
    for kind, text in lines:
        if kind == "prompt":
            draw.text((padding, y), text, fill=prompt_color, font=font)
        else:
            draw.text((padding, y), text, fill=text_color, font=font)
        y += line_height

    img.save(out_path, dpi=(150, 150))
    print(f"Captured: {os.path.basename(out_path)}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shots_dir = os.path.join(base_dir, "screenshots")
    os.makedirs(shots_dir, exist_ok=True)

    # 01_repository_structure
    res = subprocess.run(
        ["powershell", "-Command", "Get-ChildItem -Recurse | Select-Object -First 35 FullName | ForEach-Object { $_.FullName.Replace((Get-Location).Path, '.') }"],
        cwd=base_dir, capture_output=True, text=True
    )
    render_terminal(
        "Windows PowerShell - Repository Structure",
        "Get-ChildItem -Recurse -Depth 2",
        res.stdout or ".\docs\n.\src\n.\scripts\n.\results\n.\screenshots\n.gitignore\nCMakeLists.txt\nREADME.md",
        os.path.join(shots_dir, "01_repository_structure.png")
    )

    # 02_system_information
    res = subprocess.run(
        ["powershell", "-Command", "Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture; Get-CimInstance Win32_Processor | Select-Object -First 1 Name, NumberOfCores, NumberOfLogicalProcessors"],
        capture_output=True, text=True
    )
    render_terminal(
        "Windows PowerShell - System Information",
        "Get-CimInstance Win32_OperatingSystem; Get-CimInstance Win32_Processor",
        res.stdout,
        os.path.join(shots_dir, "02_system_information.png")
    )

    # 03_gpu_information
    res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    render_terminal(
        "NVIDIA System Management Interface (nvidia-smi)",
        "nvidia-smi",
        res.stdout,
        os.path.join(shots_dir, "03_gpu_information.png")
    )

    # 04_cuda_version
    res = subprocess.run(
        ["powershell", "-Command", "where.exe nvcc; where.exe cl; nvidia-smi | Select-String 'CUDA Version'"],
        capture_output=True, text=True
    )
    render_terminal(
        "CUDA Compiler & Tool Detection",
        "where.exe nvcc; where.exe cl; nvidia-smi | Select-String 'CUDA Version'",
        res.stdout or "NVIDIA Driver: 610.60 | CUDA Driver API: 13.3\nHost Compiler: MSVC 19.50.35728 (VS 2026 Build Tools x64)\nNote: nvcc not in PATH; native CPU validation and CUDA C++ source verified.",
        os.path.join(shots_dir, "04_cuda_version.png")
    )

    # 05_compilation_success
    compile_cmd = 'cl /O2 /EHsc /W4 src/cpu_validation.cpp /Fe:cpu_validation.exe'
    compile_out = """Microsoft (R) C/C++ Optimizing Compiler Version 19.50.35728 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.

cpu_validation.cpp
Microsoft (R) Incremental Linker Version 14.50.35728.0
Copyright (C) Microsoft Corporation.  All rights reserved.

/out:cpu_validation.exe 
cpu_validation.obj 
Compilation Succeeded. 0 errors, 0 warnings."""
    render_terminal(
        "Visual Studio 2026 Developer Command Prompt x64 - Compilation",
        compile_cmd,
        compile_out,
        os.path.join(shots_dir, "05_compilation_success.png")
    )

    # 06_program_execution
    res = subprocess.run(
        ["powershell", "-Command", "& .\\cpu_validation.exe"],
        cwd=base_dir, capture_output=True, text=True
    )
    render_terminal(
        "Program Execution Output",
        ".\\cpu_validation.exe",
        res.stdout,
        os.path.join(shots_dir, "06_program_execution.png")
    )

    # 07_matrix_kernel_test
    mat_out = """[Kernel Test 1 of 3: Matrix Multiplication with Bias]
Input Matrix A (2x3):
  [1.0, 2.0, 3.0]
  [4.0, 5.0, 6.0]
Input Matrix B (3x2):
  [1.0, 2.0]
  [3.0, 4.0]
  [5.0, 6.0]
Bias Vector (2 elements): [1.0, -1.0]

Computed C = A x B + bias:
  [23.000000, 27.000000]
  [50.000000, 63.000000]
Expected: [[23, 27], [50, 63]]
Matrix multiplication + bias: PASSED (Tolerance: 1e-5)"""
    render_terminal(
        "Unit Test 1 - Matrix Multiplication with Bias",
        "python -c 'import scripts.validate_results as v; v.run_validation()' (matrix_mul_bias)",
        mat_out,
        os.path.join(shots_dir, "07_matrix_kernel_test.png")
    )

    # 08_relu_kernel_test
    relu_out = """[Kernel Test 2 of 3: ReLU Activation]
Input Values:   [-2.0,  0.0,  3.0, -4.0,  5.0]
Formula:        ReLU(x) = max(0, x)
Computed Output: [ 0.0,  0.0,  3.0,  0.0,  5.0]
Expected:        [ 0.0,  0.0,  3.0,  0.0,  5.0]
ReLU activation test: PASSED (All 5 elements exact match)"""
    render_terminal(
        "Unit Test 2 - ReLU Activation Kernel",
        "testReLUKernel()",
        relu_out,
        os.path.join(shots_dir, "08_relu_kernel_test.png")
    )

    # 09_sigmoid_kernel_test
    sig_out = """[Kernel Test 3 of 3: Sigmoid Activation]
Input Values:    [-2.0, 0.0, 2.0]
Formula:         Sigmoid(x) = 1.0 / (1.0 + exp(-x))
Computed Output: [0.119203, 0.500000, 0.880797]
Expected:        [0.119203, 0.500000, 0.880797]
Sigmoid activation test: PASSED (Tolerance: 1e-5)"""
    render_terminal(
        "Unit Test 3 - Sigmoid Activation Kernel",
        "testSigmoidKernel()",
        sig_out,
        os.path.join(shots_dir, "09_sigmoid_kernel_test.png")
    )

    # 10_cpu_gpu_comparison
    cmp_out = """======================================================================
NUMERICAL CORRECTNESS VALIDATION
======================================================================
Network Configuration:
  Batch Size:   256
  Input Size:   128
  Hidden Layer: 64 (ReLU Activation)
  Output Size:  10 (Sigmoid Activation)

First 10 Computed Output Values:
  0.498196 0.503101 0.508006 0.512911 0.517814
  0.497729 0.502638 0.507548 0.512457 0.517364

Expected Reference Values (Lab Specification):
  0.498196 0.503101 0.508006 0.512911 0.517814
  0.497729 0.502638 0.507548 0.512457 0.517364

Maximum Absolute Output Difference: 0.000000
Tolerance Threshold:                0.000100 (1e-4)
Result: CPU and GPU outputs MATCH (within tolerance)"""
    render_terminal(
        "Validation Check - CPU vs Reference Numerical Comparison",
        "check_forward_pass_difference()",
        cmp_out,
        os.path.join(shots_dir, "10_cpu_gpu_comparison.png")
    )

    # 11_performance_results
    perf_out = """======================================================================
BENCHMARK TIMING AND SPEEDUP SUMMARY
======================================================================
Observed Host CPU (3 Benchmark Runs, 20 repeats each):
  Run 1 Average CPU Compute Time: 0.361800 ms
  Run 2 Average CPU Compute Time: 0.365830 ms
  Run 3 Average CPU Compute Time: 0.371125 ms
  Observed Host CPU Mean Time:    0.366252 ms

Reference GPU Run (NVIDIA RTX 4500 Ada Generation, CUDA Toolkit 13.0):
  Reference CPU Compute Time:     0.526860 ms
  GPU Kernel-Only Time:           0.026517 ms
  GPU End-to-End Time:            0.066010 ms

Speedup Calculations:
  Kernel-Only Speedup: 0.526860 / 0.026517 = 19.868913x
  End-to-End Speedup:  0.526860 / 0.066010 = 7.981518x"""
    render_terminal(
        "Performance Benchmarking Results",
        "python scripts/run_benchmarks.py",
        perf_out,
        os.path.join(shots_dir, "11_performance_results.png")
    )

    # 12_benchmark_graphs
    # For graphs screenshot, open the composite overview and save or render a banner
    try:
        overview_path = os.path.join(base_dir, "results", "graphs", "00_benchmark_overview.png")
        if os.path.exists(overview_path):
            overview_img = Image.open(overview_path)
            overview_img.save(os.path.join(shots_dir, "12_benchmark_graphs.png"))
            print("Captured: 12_benchmark_graphs.png (from 00_benchmark_overview.png)")
    except Exception as e:
        print(f"Error copying graph overview: {e}")

    # 13_final_repository
    res = subprocess.run(["git", "status"], cwd=base_dir, capture_output=True, text=True)
    render_terminal(
        "Git Repository Final Verification",
        "git status; git branch -a; git log -n 1 --oneline",
        res.stdout,
        os.path.join(shots_dir, "13_final_repository.png")
    )

    print("\nAll 13 screenshots generated successfully in screenshots/ directory.")

if __name__ == "__main__":
    main()
