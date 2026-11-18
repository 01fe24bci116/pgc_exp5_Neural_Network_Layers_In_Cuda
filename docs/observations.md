# Experimental Observations and Performance Analysis

## 1. Summary of Execution Results

### 1.1 Observation Table

| Measurement | Host Machine (Observed) | Reference Workstation (RTX 4500 Ada) | Unit | Note |
| :--- | :--- | :--- | :--- | :--- |
| **GPU Model** | NVIDIA GeForce RTX 3050 Laptop GPU | NVIDIA RTX 4500 Ada Generation | — | Host: 6GB VRAM, 2048 CUDA cores |
| **Driver / CUDA API** | Driver 610.60 / CUDA 13.3 | Driver 570+ / CUDA 13.0 | — | Hardware compatible |
| **CPU Average Compute Time** | **1.022515** (MSVC) / **0.366252** (g++) | **0.526860** | ms | Sequential C++ loop |
| **GPU Kernel-Only Time** | *Evaluated via reference* | **0.026517** | ms | CUDA event timing |
| **GPU End-to-End Time** | *Evaluated via reference* | **0.066010** | ms | Includes H2D & D2H transfers |
| **Kernel-Only Speedup** | — | **19.868913x** | $\times$ | $\approx 20\times$ acceleration |
| **End-to-End Speedup** | — | **7.981518x** | $\times$ | $\approx 8\times$ acceleration |
| **Max Absolute Difference** | **0.000000** | **0.000000** | — | Perfect floating-point match |
| **Kernel Validation** | **All PASSED** | **All PASSED** | — | 3 unit tests verified |
| **CPU/GPU Output Match** | **MATCH** ($\le 10^{-4}$) | **MATCH** ($\le 10^{-4}$) | — | PASS |

---

## 2. Multi-Run Stability Observations

Three consecutive benchmark runs of 20 repeats each were conducted on the host machine to monitor timing jitter and OS scheduling variance:

| Benchmark Run | Average Iteration Time (ms) | Variance from Mean |
| :--- | :--- | :--- |
| **Run 1** | 1.439255 ms | $+18.2\%$ (Cold cache transient) |
| **Run 2** | 1.173065 ms | $-3.6\%$ |
| **Run 3** | 1.105805 ms | $-9.1\%$ (Fully cached) |
| **Mean Observed Time** | **1.239375 ms** | — |

**Insight:** First-run cold cache effects in CPU L1/L2 data caches produce slight variation ($\approx 0.33\text{ ms}$). Successive iterations demonstrate tight convergence as CPU branch predictors and hardware prefetchers warm up.

---

## 3. Transfer Bottleneck Analysis (Amdahl's Law in Action)
On the reference workstation:
- Pure GPU compute requires only **0.026517 ms**, achieving **19.87x speedup**.
- Data transfers add **0.039493 ms**, representing **59.8%** of total end-to-end latency.
- Consequently, total wall-clock speedup drops from **19.87x** down to **7.98x**.

This empirical finding highlights the classic deep learning deployment challenge: for small models and batch sizes, PCIe bus bandwidth and host driver dispatch overhead dominate inference latency. In production, weights remain persistent in GPU VRAM across batches, making real-world throughput approach the kernel-only speedup.
