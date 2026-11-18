# Implementation Details: CUDA Kernels and Host Architectures

## 1. CUDA Kernel Architectures

### 1.1 Matrix Multiplication with Bias (`matrixMulBiasKernel`)
The matrix multiplication kernel performs a 2D grid decomposition where each CUDA thread computes exactly one scalar element of the product matrix $C \in \mathbb{R}^{M \times N}$, adding the corresponding column bias $\text{bias}[\text{col}]$.

```cuda
__global__ void matrixMulBiasKernel(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum + bias[col];
    }
}
```

#### Thread Indexing & Geometry
- **Block Geometry:** `dim3 block(16, 16);` (256 threads per thread block).
- **Grid Configuration for Layer 1:**
  $$\text{grid1.x} = \lceil \text{HIDDEN\_SIZE} / 16 \rceil = \lceil 64 / 16 \rceil = 4$$
  $$\text{grid1.y} = \lceil \text{BATCH} / 16 \rceil = \lceil 256 / 16 \rceil = 16$$
  Total blocks in Grid 1 = $4 \times 16 = 64$ blocks (16,384 threads), exactly matching the output matrix size ($256 \times 64$).
- **Grid Configuration for Layer 2:**
  $$\text{grid2.x} = \lceil \text{OUTPUT\_SIZE} / 16 \rceil = \lceil 10 / 16 \rceil = 1$$
  $$\text{grid2.y} = \lceil \text{BATCH} / 16 \rceil = \lceil 256 / 16 \rceil = 16$$
  Total blocks in Grid 2 = $1 \times 16 = 16$ blocks (4,096 threads).
- **Boundary Checks:** Because `OUTPUT_SIZE` (10) is less than `block.x` (16), threads where $\text{col} \ge 10$ are guarded by `if (row < M && col < N)` to prevent illegal memory writes.

---

### 1.2 ReLU Activation Kernel (`reluKernel`)
The ReLU kernel maps an array of values in device global memory in-place, clamping negative values to zero.

```cuda
__global__ void reluKernel(float* values, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        values[idx] = fmaxf(0.0f, values[idx]);
    }
}
```
- **Thread Geometry:** 1D layout with `threads = 256` and $\text{blocks} = \lceil \text{count} / 256 \rceil$.
- **Intrinsic Optimization:** Utilizes `fmaxf()`, which compiles to a single native GPU assembly instruction (`FMNMX`).

---

### 1.3 Sigmoid Activation Kernel (`sigmoidKernel`)
The Sigmoid kernel applies the logistic function $S(x) = \frac{1}{1 + e^{-x}}$ in-place across the output tensor.

```cuda
__global__ void sigmoidKernel(float* values, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        values[idx] = 1.0f / (1.0f + expf(-values[idx]));
    }
}
```
- **Intrinsic Optimization:** `expf()` maps directly to the GPU's Special Function Units (SFUs), delivering multi-teraflop throughput for transcendental functions.

---

## 2. CPU Reference Implementations

To ensure strict verification, standard C++ functions duplicate the exact mathematical logic using cache-friendly row-major strides:

```cpp
void cpuMatMulBias(const std::vector<float>& A,
                   const std::vector<float>& B,
                   const std::vector<float>& bias,
                   std::vector<float>& C,
                   int M, int N, int K)
{
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum + bias[col];
        }
    }
}
```

---

## 3. Execution Pipeline and Stream Semantics

In CUDA, kernels issued on the default stream (`stream 0`) execute in strict FIFO order:
1. `matrixMulBiasKernel<<<grid1, block>>>(dX, dW1, db1, dZ1, ...)`
2. `reluKernel<<<blocks1, 256>>>(dZ1, ...)`
3. `matrixMulBiasKernel<<<grid2, block>>>(dZ1, dW2, db2, dZ2, ...)`
4. `sigmoidKernel<<<blocks2, 256>>>(dZ2, ...)`

Because the GPU driver serializes operations in the default stream, Layer 2 kernels are guaranteed to read the fully updated activations produced by Layer 1 without manual inter-kernel synchronization.
