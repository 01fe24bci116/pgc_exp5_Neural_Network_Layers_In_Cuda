// Experiment 5: Neural Network Layers in CUDA - CPU Validation Harness
// Validates mathematical correctness of the neural network layers,
// individual kernel test logic, and measures native CPU execution benchmarks.

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstdlib>

constexpr int BATCH       = 256;
constexpr int INPUT_SIZE  = 128;
constexpr int HIDDEN_SIZE = 64;
constexpr int OUTPUT_SIZE = 10;
constexpr int REPEATS     = 20;

bool nearlyEqual(float a, float b, float tolerance = 1e-5f)
{
    return std::fabs(a - b) <= tolerance;
}

// -------------------------------------------------------------
// CPU Implementation of Kernels & Layers (Reference Specification)
// -------------------------------------------------------------

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

void cpuReLU(std::vector<float>& values)
{
    for (float& x : values) {
        x = std::max(0.0f, x);
    }
}

void cpuSigmoid(std::vector<float>& values)
{
    for (float& x : values) {
        x = 1.0f / (1.0f + std::exp(-x));
    }
}

void cpuForwardPass(const std::vector<float>& X,
                    const std::vector<float>& W1,
                    const std::vector<float>& b1,
                    const std::vector<float>& W2,
                    const std::vector<float>& b2,
                    std::vector<float>& output)
{
    std::vector<float> z1(BATCH * HIDDEN_SIZE);
    cpuMatMulBias(X, W1, b1, z1, BATCH, HIDDEN_SIZE, INPUT_SIZE);
    cpuReLU(z1);

    std::vector<float> z2(BATCH * OUTPUT_SIZE);
    cpuMatMulBias(z1, W2, b2, z2, BATCH, OUTPUT_SIZE, HIDDEN_SIZE);
    cpuSigmoid(z2);

    output = z2;
}

// -------------------------------------------------------------
// Unit Tests for Kernels (Section 8, Step 2)
// -------------------------------------------------------------

bool testMatrixKernelCpu()
{
    // A is 2x3, B is 3x2. Expected A*B = [22,28;49,64].
    // Add bias [1,-1] => [23,27;50,63].
    std::vector<float> hA = {1, 2, 3, 4, 5, 6};
    std::vector<float> hB = {1, 2, 3, 4, 5, 6};
    std::vector<float> hBias = {1, -1};
    std::vector<float> hC(4, 0.0f);

    cpuMatMulBias(hA, hB, hBias, hC, 2, 2, 3);

    bool ok = nearlyEqual(hC[0], 23.0f) && nearlyEqual(hC[1], 27.0f) &&
              nearlyEqual(hC[2], 50.0f) && nearlyEqual(hC[3], 63.0f);
    return ok;
}

bool testReLUKernelCpu()
{
    std::vector<float> h = {-2.0f, 0.0f, 3.0f, -4.0f, 5.0f};
    cpuReLU(h);

    bool ok = nearlyEqual(h[0], 0.0f) && nearlyEqual(h[1], 0.0f) &&
              nearlyEqual(h[2], 3.0f) && nearlyEqual(h[3], 0.0f) &&
              nearlyEqual(h[4], 5.0f);
    return ok;
}

bool testSigmoidKernelCpu()
{
    std::vector<float> h = {-2.0f, 0.0f, 2.0f};
    cpuSigmoid(h);

    bool ok = nearlyEqual(h[0], 1.0f / (1.0f + std::exp(2.0f))) &&
              nearlyEqual(h[1], 0.5f) &&
              nearlyEqual(h[2], 1.0f / (1.0f + std::exp(-2.0f)));
    return ok;
}

int main()
{
    std::cout << "======================================================================\n";
    std::cout << "EXPERIMENT 5: NEURAL NETWORK LAYERS IN CUDA - VALIDATION HARNESS\n";
    std::cout << "Course: Cloud / GPU Computing Laboratory\n";
    std::cout << "======================================================================\n\n";

    std::cout << "Testing individual CUDA kernel logic...\n";
    bool matrixOK  = testMatrixKernelCpu();
    bool reluOK    = testReLUKernelCpu();
    bool sigmoidOK = testSigmoidKernelCpu();

    std::cout << "Matrix multiplication + bias: " << (matrixOK  ? "PASSED" : "FAILED") << "\n";
    std::cout << "ReLU:                         " << (reluOK    ? "PASSED" : "FAILED") << "\n";
    std::cout << "Sigmoid:                      " << (sigmoidOK ? "PASSED" : "FAILED") << "\n";

    if (!(matrixOK && reluOK && sigmoidOK)) {
        std::cerr << "A kernel test failed. Stop and debug before continuing.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nInitializing deterministic inputs and weights...\n";
    std::vector<float> X(BATCH * INPUT_SIZE);
    std::vector<float> W1(INPUT_SIZE * HIDDEN_SIZE);
    std::vector<float> b1(HIDDEN_SIZE);
    std::vector<float> W2(HIDDEN_SIZE * OUTPUT_SIZE);
    std::vector<float> b2(OUTPUT_SIZE);

    for (size_t i = 0; i < X.size(); ++i)  X[i]  = 0.5f * std::sin(static_cast<float>(i) * 0.013f);
    for (size_t i = 0; i < W1.size(); ++i) W1[i] = 0.08f * std::cos(static_cast<float>(i) * 0.017f);
    for (size_t i = 0; i < b1.size(); ++i) b1[i] = 0.01f * static_cast<float>(i % 7);
    for (size_t i = 0; i < W2.size(); ++i) W2[i] = 0.10f * std::sin(static_cast<float>(i) * 0.019f);
    for (size_t i = 0; i < b2.size(); ++i) b2[i] = 0.02f * static_cast<float>(i % 5);

    std::cout << "\nRunning CPU forward pass benchmarking (" << REPEATS << " iterations)...\n";
    std::vector<float> cpuOutput(BATCH * OUTPUT_SIZE);

    // Warm-up iteration
    cpuForwardPass(X, W1, b1, W2, b2, cpuOutput);

    auto cpuStart = std::chrono::steady_clock::now();
    for (int r = 0; r < REPEATS; ++r) {
        cpuForwardPass(X, W1, b1, W2, b2, cpuOutput);
    }
    auto cpuStop = std::chrono::steady_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count() / REPEATS;

    std::vector<float> expectedFirst10 = {
        0.498196f, 0.503101f, 0.508006f, 0.512911f, 0.517814f,
        0.497729f, 0.502638f, 0.507548f, 0.512457f, 0.517364f
    };

    float maxDifference = 0.0f;
    for (size_t i = 0; i < 10; ++i) {
        maxDifference = std::max(maxDifference, std::fabs(cpuOutput[i] - expectedFirst10[i]));
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nForward-pass dimensions:\n";
    std::cout << "Input:        " << BATCH << " x " << INPUT_SIZE << "\n";
    std::cout << "Hidden layer: " << BATCH << " x " << HIDDEN_SIZE << "\n";
    std::cout << "Output:       " << BATCH << " x " << OUTPUT_SIZE << "\n";

    std::cout << "\nAverage CPU compute time: " << cpuMs << " ms\n";
    std::cout << "Maximum absolute difference against lab reference: " << maxDifference << "\n";
    std::cout << "Mathematical consistency check: "
              << (maxDifference < 1e-4f ? "MATCH (within tolerance)" : "DO NOT MATCH")
              << "\n";

    std::cout << "\nFirst 10 output values:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << cpuOutput[i] << (i == 9 ? '\n' : ' ');
    }

    std::cout << "\nExpected reference values:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << expectedFirst10[i] << (i == 9 ? '\n' : ' ');
    }

    return (maxDifference < 1e-4f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
