# Troubleshooting and Environment Diagnostic Guide

This guide addresses common configuration, compilation, and runtime issues encountered when compiling and running CUDA neural network experiments on Windows.

---

## 1. `nvcc fatal : Cannot find compiler 'cl.exe' in PATH`

### Problem Description
The NVIDIA CUDA compiler (`nvcc`) is a split-compiler: it preprocesses device code and emits PTX/machine code, while delegating host C++ code to the platform's native C++ compiler (Microsoft Visual C++ `cl.exe` on Windows). If `cl.exe` is not present in the current shell's `PATH` variable, `nvcc` aborts immediately.

### Diagnostic Command
```cmd
where cl
```

### Resolution
1. Open the **x64 Native Tools Command Prompt for VS 2022** (or VS 2026 Build Tools) from the Windows Start Menu.
2. Alternatively, invoke the environment initialization script within standard CMD:
   ```cmd
   call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
   ```
3. If Visual Studio is not installed, install the **Desktop development with C++** workload via the Visual Studio Installer, ensuring that **MSVC v143** and the **Windows 10/11 SDK** are selected.

---

## 2. `nvcc is not recognized as an internal or external command`

### Problem Description
The CUDA Toolkit binary directory has not been added to system environment variables.

### Diagnostic Command
```cmd
where.exe nvcc
```

### Resolution
1. Verify the CUDA Toolkit installation path (typically `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin`).
2. Add the path to system `PATH`:
   ```powershell
   [System.Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin", "User")
   ```
3. Restart the terminal.

---

## 3. `LINK : fatal error LNK1104: cannot open file 'neural_network.exe'`

### Cause & Resolution
- **Cause 1: Process Locking.** An earlier instance of `neural_network.exe` is still actively running or paused. Use Windows Task Manager or run `taskkill /f /im neural_network.exe` before recompiling.
- **Cause 2: Protected Directory Permissions.** Compiling inside write-protected directories such as `C:\Windows\System32` triggers access denial. Always build within a dedicated user directory.

---

## 4. `No CUDA-capable NVIDIA GPU was detected`

### Diagnostic Workflow
1. Run `nvidia-smi` to verify that the GPU is detected by the operating system kernel driver.
2. Check Windows Device Manager under **Display Adapters** to confirm that the NVIDIA graphics card is enabled and functional.
3. Verify that the installed NVIDIA graphics driver supports the targeted CUDA Toolkit version.

---

## 5. Kernel Test Failures or Numerical Mismatch

### Diagnostic Checklist
- **Thread Matrix Strides:** Verify that 2D indexing matches row-major ordering: `C[row * N + col]`.
- **Matrix Dimension Arguments:** Ensure the order of parameters (`M, N, K`) matches between host loops and GPU kernel signatures.
- **Floating-Point Tolerances:** Intermediate accumulation orders can introduce slight round-off variations between CPU and GPU FMA instructions. Discrepancies within $10^{-5}$ are normal; deviations greater than $10^{-4}$ indicate algorithmic errors.
