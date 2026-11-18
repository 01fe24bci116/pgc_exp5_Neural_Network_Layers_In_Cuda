# Screenshots and Execution Evidence Index

This directory maintains execution evidence, environment verification logs, and benchmark visualizations for **Experiment 5: Neural Network Layers in CUDA**.

---

## Index of Verification Screenshots

| File Name | Description & Evidence Provided |
| :--- | :--- |
| `01_repository_structure.png` | Verifies full project tree, including source, documentation, test scripts, and results directories. |
| `02_system_information.png` | Captures host OS specifications (Windows 11 x64) and processor topology. |
| `03_gpu_information.png` | Displays live `nvidia-smi` output detailing the NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, Driver 610.60). |
| `04_cuda_version.png` | Documents CUDA runtime / driver API level (CUDA 13.3) and host toolchain detection. |
| `05_compilation_success.png` | Confirms clean compilation under MSVC (`cl.exe /O2 /EHsc /W4`) with zero warnings and zero errors. |
| `06_program_execution.png` | Shows full forward-pass validation output, tensor dimensions, and timing output. |
| `07_matrix_kernel_test.png` | Validates Matrix Multiplication with Bias unit test ($A \times B + \text{bias} = [[23, 27], [50, 63]]$) passing with tolerance $\le 10^{-5}$. |
| `08_relu_kernel_test.png` | Validates ReLU activation unit test ($[-2, 0, 3, -4, 5] \to [0, 0, 3, 0, 5]$). |
| `09_sigmoid_kernel_test.png` | Validates Sigmoid activation unit test ($[-2, 0, 2] \to [0.119203, 0.5, 0.880797]$). |
| `10_cpu_gpu_comparison.png` | Validates numerical consistency across all outputs, confirming a maximum absolute difference $\le 10^{-4}$ and `MATCH (within tolerance)`. |
| `11_performance_results.png` | Summarizes multi-run benchmark measurements, CPU timings, and speedup analysis against reference GPU hardware. |
| `12_benchmark_graphs.png` | Composite four-panel figure displaying execution latency comparisons, speedup scaling, and transfer overhead breakdown. |
| `13_final_repository.png` | Demonstrates clean git status, branch tracking, and commit preparation prior to deployment. |

---

## Reproducibility Note

All screenshots are generated directly from system utilities (`nvidia-smi`, `cl.exe`, PowerShell) and the test suites in `scripts/`. They can be regenerated at any time using:

```cmd
python scripts/capture_screenshots.py
```
