# Performance Benchmarking Methodology

## 1. Benchmarking Objectives
Accurate performance analysis on heterogeneous CPU-GPU systems requires separating hardware compute throughput from interconnect communication latency. This experiment measures two distinct timing scopes:
1. **Kernel-Only Latency ($T_{\text{GPU\_kernel}}$):** Raw GPU arithmetic execution time with inputs resident in device memory.
2. **End-to-End Latency ($T_{\text{GPU\_e2e}}$):** Total round-trip inference latency including Host-to-Device data transfers, kernel execution, device synchronization, and Device-to-Host transfer.

---

## 2. Timing Instrumentation

### 2.1 GPU Kernel-Only Timing: CUDA Events
CUDA kernels execute asynchronously on the GPU device relative to the host CPU. Standard CPU clocks cannot measure kernel durations without imposing artificial synchronization barriers that distort streaming queues.

CUDA Events measure duration on the GPU hardware clock:
```cuda
cudaEvent_t eventStart, eventStop;
CUDA_CHECK(cudaEventCreate(&eventStart));
CUDA_CHECK(cudaEventCreate(&eventStop));

CUDA_CHECK(cudaEventRecord(eventStart));
for (int r = 0; r < REPEATS; ++r) {
    launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);
}
CUDA_CHECK(cudaEventRecord(eventStop));
CUDA_CHECK(cudaEventSynchronize(eventStop));

float gpuKernelMs = 0.0f;
CUDA_CHECK(cudaEventElapsedTime(&gpuKernelMs, eventStart, eventStop));
gpuKernelMs /= REPEATS;
```

### 2.2 End-to-End Latency: Monotonic Host Clock
End-to-end measurement encompasses data transfers across the PCIe bus and host driver coordination. This scope is recorded using C++ `std::chrono::steady_clock`:
```cpp
auto gpuTotalStart = std::chrono::steady_clock::now();
for (int r = 0; r < REPEATS; ++r) {
    // 1. Host-to-Device copies (Inputs, Weights, Biases)
    cudaMemcpy(dX, ...);
    cudaMemcpy(dW1, ...);
    cudaMemcpy(db1, ...);
    cudaMemcpy(dW2, ...);
    cudaMemcpy(db2, ...);

    // 2. Compute
    launchGpuForwardPass(...);
    cudaDeviceSynchronize();

    // 3. Device-to-Host copy (Predictions)
    cudaMemcpy(gpuOutput.data(), dZ2, ...);
}
auto gpuTotalStop = std::chrono::steady_clock::now();
double gpuEndToEndMs = std::chrono::duration<double, std::milli>(gpuTotalStop - gpuTotalStart).count() / REPEATS;
```

---

## 3. GPU Warm-up Phase
Upon initial invocation, CUDA programs encounter one-time lazy initialization latencies:
- Driver context creation on the target device
- Dynamic linking of device runtime symbols
- GPU power-state transitions from idle ($P_8$) to active boost ($P_0$)
- Creation of internal stream queues and page tables

To eliminate these one-time transients, an untimed **warm-up forward pass** executes and synchronizes prior to benchmark recording:
```cuda
// Warm-up pass
launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);
CUDA_CHECK(cudaDeviceSynchronize());
```

---

## 4. Analytical Metrics and Speedup Formulation

### 4.1 Kernel-Only Speedup
Measures computational throughput scaling:
$$S_{\text{kernel}} = \frac{T_{\text{CPU\_compute}}}{T_{\text{GPU\_kernel}}}$$

### 4.2 End-to-End Speedup
Measures system-level user-perceived speedup:
$$S_{\text{e2e}} = \frac{T_{\text{CPU\_compute}}}{T_{\text{GPU\_e2e}}}$$

### 4.3 Transfer Overhead Analysis
The fraction of end-to-end execution consumed by PCIe bus traffic:
$$\text{Transfer Overhead} = T_{\text{GPU\_e2e}} - T_{\text{GPU\_kernel}}$$
$$\% \text{ Transfer Time} = \left( \frac{T_{\text{GPU\_e2e}} - T_{\text{GPU\_kernel}}}{T_{\text{GPU\_e2e}}} \right) \times 100\%$$
