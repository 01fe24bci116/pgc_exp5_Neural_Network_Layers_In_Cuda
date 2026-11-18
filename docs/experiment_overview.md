# Experiment Overview: Neural Network Layers in CUDA

## 1. Experiment Aim
The primary objective of this experiment is to design, implement, and evaluate custom CUDA C++ kernels that execute fundamental deep learning layers:
- 2D General Matrix Multiplication with row/column bias addition ($\text{GEMM} + \text{Bias}$)
- Rectified Linear Unit (ReLU) elementwise activation
- Sigmoid ($\sigma$) elementwise activation

These kernels are composed into a complete two-layer feed-forward neural network forward inference pipeline. An identical CPU reference pipeline is implemented in standard C++ to evaluate numerical correctness, verify tolerances, and assess GPU speedup over sequential CPU execution.

---

## 2. Learning Outcomes
Upon completing this laboratory module, developers and researchers can:
1. Explain thread hierarchy abstractions in CUDA, specifically grid dimensions, thread block dimensions, and global thread index computation.
2. Formulate 2D tile-mapped matrix multiplication kernels with bias accumulation, incorporating boundary condition checks.
3. Formulate 1D parallel elementwise activation kernels (ReLU and Sigmoid) using hardware-accelerated math intrinsics (`fmaxf`, `expf`).
4. Manage CUDA memory lifecycles (`cudaMalloc`, `cudaMemcpy`, `cudaFree`) between Host (CPU DRAM) and Device (GPU VRAM).
5. Conduct numerical correctness validation using floating-point tolerance thresholds ($\le 10^{-4}$).
6. Measure execution performance using high-resolution CUDA events and host monotonic clocks, distinguishing compute latency from PCIe transfer bottlenecks.

---

## 3. Network Architecture & Mathematical Formulations
The experiment evaluates forward inference across a two-layer multi-layer perceptron (MLP). There is no backpropagation, gradient calculation, or weight optimization; evaluation focuses strictly on inference throughput.

### 3.1 Network Topology
```
Input Tensor X          Hidden Layer (Z1, A1)          Output Layer (Z2, Y)
[256 x 128]      --->       [256 x 64]          --->        [256 x 10]
              (W1: 128x64, b1: 64)           (W2: 64x10, b2: 10)
```

### 3.2 Mathematical Formulation
The feed-forward computation sequence proceeds as follows:

1. **Hidden Layer Pre-activation ($Z_1$):**
   $$Z_1 = X \times W_1 + b_1$$
   Where $X \in \mathbb{R}^{256 \times 128}$, $W_1 \in \mathbb{R}^{128 \times 64}$, and $b_1 \in \mathbb{R}^{64}$. Each output element $(i, j)$ is computed as:
   $$Z_1[i, j] = \left( \sum_{k=0}^{127} X[i, k] \cdot W_1[k, j] \right) + b_1[j]$$

2. **Hidden Layer Activation ($A_1$):**
   $$A_1 = \text{ReLU}(Z_1) = \max(0, Z_1)$$

3. **Output Layer Pre-activation ($Z_2$):**
   $$Z_2 = A_1 \times W_2 + b_2$$
   Where $A_1 \in \mathbb{R}^{256 \times 64}$, $W_2 \in \mathbb{R}^{64 \times 10}$, and $b_2 \in \mathbb{R}^{10}$:
   $$Z_2[i, j] = \left( \sum_{k=0}^{63} A_1[i, k] \cdot W_2[k, j] \right) + b_2[j]$$

4. **Output Activation ($Y$):**
   $$Y = \text{Sigmoid}(Z_2) = \frac{1}{1 + e^{-Z_2}}$$

---

## 4. Tensor Dimensions and Memory Allocation

| Tensor Symbol | Description | Dimensions | Number of Float32 Elements | Memory Footprint (Bytes) |
| :--- | :--- | :--- | :--- | :--- |
| **X** | Input batch | $256 \times 128$ | 32,768 | 131,072 B (128 KB) |
| **W1** | Layer 1 weight matrix | $128 \times 64$ | 8,192 | 32,768 B (32 KB) |
| **b1** | Layer 1 bias vector | $64$ | 64 | 256 B |
| **Z1 / A1** | Hidden layer intermediate | $256 \times 64$ | 16,384 | 65,536 B (64 KB) |
| **W2** | Layer 2 weight matrix | $64 \times 10$ | 640 | 2,560 B (2.5 KB) |
| **b2** | Layer 2 bias vector | $10$ | 10 | 40 B |
| **Z2 / Y** | Output predictions | $256 \times 10$ | 2,560 | 10,240 B (10 KB) |

**Total Model & Activation Working Set:** $\approx 242.7\text{ KB}$. This compact memory footprint enables analysis of memory transfer latency relative to pure compute performance on modern GPUs.
