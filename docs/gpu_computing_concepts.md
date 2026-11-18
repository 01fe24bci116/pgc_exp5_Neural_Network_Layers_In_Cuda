# GPU Computing and Parallel Inference Architecture

## Core Concepts in Parallel Kernel Execution

### 1. SIMT Execution Paradigm
Modern graphics processors utilize Single Instruction, Multiple Threads (SIMT). Execution threads are scheduled in lockstep units called warps (32 threads). When designing parallel feed-forward layers, organizing workloads into contiguous 1D or 2D thread blocks ensures coalesced global memory access and balanced load across Streaming Multiprocessors.

### 2. Affine Transformations and Kernel Fusion
In dense artificial neural network architectures, matrix multiplication followed by bias accumulation is traditionally expressed as:
$$Z = X W + b$$
Fusing the additive bias directly inside the inner product kernel avoids additional kernel launch overhead and eliminates an extra memory read/write pass over global device memory.

### 3. Activation Functions and Device Intrinsics
Non-linear activation functions such as ReLU ($\max(0, x)$) and Sigmoid ($\frac{1}{1 + e^{-x}}$) can be evaluated elementwise. High-performance implementations utilize device intrinsics that compile down directly to specialized hardware instructions in the GPU ALU and Special Function Units (SFU).

### 4. Memory Hierarchy and Latency Hiding
High computational speedup is achieved when memory transfers between host DRAM and device VRAM are minimized. When evaluating throughput, distinguishing between raw device kernel time (measured via hardware timestamps) and complete round-trip wall-clock time highlights the impact of PCIe bus transmission on end-to-end performance.
