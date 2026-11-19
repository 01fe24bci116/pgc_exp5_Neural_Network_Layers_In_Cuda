# Viva Voce Questions & Answers: Neural Network Layers in CUDA

This guide provides a comprehensive preparation resource for technical evaluations, viva examinations, and lab assessments regarding **CUDA Parallel Programming and Neural Network Layer Acceleration**.

---

## 1. CUDA Fundamentals & Thread Hierarchy

### Q1: Explain the CUDA thread hierarchy (Threads, Blocks, Grids).
**Answer:**
CUDA organizes parallel execution units hierarchically:
- **Thread:** The fundamental execution unit mapped to a CUDA core. Each thread possesses private local registers, a program counter, and execution state.
- **Thread Block:** A cooperative group of threads executed together on a Streaming Multiprocessor (SM). Threads within a block can synchronize using `__syncthreads()` and share low-latency on-chip shared memory.
- **Grid:** A collection of thread blocks launched by a single kernel invocation (`<<<grid, block>>>`). Blocks within a grid execute independently and in any order.

### Q2: How is the global 2D thread index computed in `matrixMulBiasKernel`?
**Answer:**
For a 2D grid of 2D thread blocks:
$$\text{row} = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}$$
$$\text{col} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
This computes the row and column coordinates in the destination matrix $C[M \times N]$. Memory address offset is calculated as `row * N + col` using row-major layout.

### Q3: Why is a boundary check `if (row < M && col < N)` necessary in CUDA kernels?
**Answer:**
Grid dimensions are calculated by rounding up:
$$\text{gridDim} = \lceil \text{size} / \text{blockDim} \rceil = (\text{size} + \text{blockDim} - 1) / \text{blockDim}$$
When the problem dimension is not an exact multiple of the block size (such as $\text{OUTPUT\_SIZE} = 10$ with $\text{blockDim.x} = 16$), surplus threads are scheduled in the boundary blocks. The boundary condition prevents those threads from reading or writing outside allocated buffer bounds, preventing memory corruption or segmentation faults.

---

## 2. Neural Network Forward Inference Kernels

### Q4: What layers are implemented in this CUDA experiment?
**Answer:**
The experiment implements a two-layer feed-forward Multi-Layer Perceptron (MLP) forward pass:
1. **Layer 1 GEMM + Bias:** $Z_1 = X \times W_1 + b_1$ (dimensions: $[256 \times 128] \times [128 \times 64] + [64] \to [256 \times 64]$)
2. **Layer 1 Activation (ReLU):** $A_1 = \max(0, Z_1)$ (in-place $[256 \times 64]$)
3. **Layer 2 GEMM + Bias:** $Z_2 = A_1 \times W_2 + b_2$ (dimensions: $[256 \times 64] \times [64 \times 10] + [10] \to [256 \times 10]$)
4. **Layer 2 Activation (Sigmoid):** $Y = \frac{1}{1 + e^{-Z_2}}$ (in-place $[256 \times 10]$)

### Q5: Why is activation applied in-place?
**Answer:**
In-place activation transforms pre-activation tensor $Z$ directly in global device memory:
- Eliminates additional device memory allocations (`cudaMalloc`).
- Conserves GPU memory bandwidth by avoiding separate load and store streams across distinct memory regions.
- Maintains cache locality within the L2 cache for subsequent kernel launches.

### Q6: What CUDA hardware math intrinsics are used for activations, and why?
**Answer:**
- `fmaxf(0.0f, x)` is used for ReLU, mapped directly to hardware floating-point maximum instructions without conditional branching, avoiding warp divergence.
- `expf(-x)` is used for Sigmoid calculation $\frac{1}{1 + \exp(-x)}$, executing via GPU Special Function Units (SFUs) for high single-precision throughput.

---

## 3. Memory Architecture & PCIe Transfers

### Q7: Explain the difference between Kernel-Only timing and End-to-End timing.
**Answer:**
- **Kernel-Only Timing:** Measures strictly the time the GPU spends executing compute kernels (`matrixMulBiasKernel`, `reluKernel`, `sigmoidKernel`). In this experiment, data already resides in GPU VRAM, yielding maximum hardware acceleration (~32.1× speedup).
- **End-to-End Timing:** Measures complete inference time, including:
  1. Host-to-Device transfer (`cudaMemcpyHostToDevice`) of input $X$ and weights.
  2. Device kernel execution.
  3. Device synchronization (`cudaDeviceSynchronize()`).
  4. Device-to-Host transfer (`cudaMemcpyDeviceToHost`) of output predictions.
  Because PCIe bandwidth (~16–32 GB/s) is orders of magnitude slower than GPU VRAM bandwidth (>200 GB/s), PCIe transfer overhead reduces the effective speedup to ~11.3×.

### Q8: What is CUDA unified memory vs explicit memory management?
**Answer:**
- **Explicit Management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`):** The developer explicitly allocates device memory and manages transfers across the PCIe bus between CPU Host and GPU Device. Gives optimal fine-grained control over transfer timing and overlap.
- **Unified Memory (`cudaMallocManaged`):** Provides a single coherent memory pointer accessible by both CPU and GPU. The CUDA driver and OS page fault engine migrate pages on-demand across the interconnect.

---

## 4. Benchmarking & Validation

### Q9: Why are CUDA Events (`cudaEvent_t`) preferred over CPU timers for GPU profiling?
**Answer:**
GPU kernel launches are asynchronous: host CPU threads queue commands into a CUDA stream and return immediately before the GPU completes execution. Using CPU wall-clock timers without explicit barriers would measure launch overhead rather than compute duration. CUDA events are recorded directly into the GPU command queue with sub-microsecond precision and properly capture device execution boundaries without CPU jitter.

### Q10: How is correctness verified between CPU and GPU implementations?
**Answer:**
Deterministic input tensors and weight matrices are initialized identically on both host and device (using sine/cosine generation). A sequential CPU forward pass computes a reference output tensor. The maximum absolute elementwise difference is computed:
$$\epsilon = \max_{i} |Y_{\text{CPU}}[i] - Y_{\text{GPU}}[i]|$$
The validation passes if $\epsilon < 10^{-4}$ (0.0001), accounting for minor floating-point associativity differences between CPU SIMD and GPU FMA (Fused Multiply-Add) execution.
