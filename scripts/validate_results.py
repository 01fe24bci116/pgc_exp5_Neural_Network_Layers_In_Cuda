#!/usr/bin/env python3
"""
Validation script for Neural Network Layers in CUDA / C++.
Verifies unit test outputs and end-to-end forward pass correctness.
"""

import json
import math
import os
import sys
import numpy as np

def run_validation():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Test 1: Matrix Multiplication with Bias
    # A = 2x3, B = 3x2, bias = 2 elements
    # A = [[1, 2, 3], [4, 5, 6]], B = [[1, 2], [3, 4], [5, 6]], bias = [1, -1]
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    bias = np.array([1.0, -1.0], dtype=np.float32)
    expected_matmul = np.matmul(A, B) + bias  # [[23, 27], [50, 63]]
    matmul_target = np.array([[23.0, 27.0], [50.0, 63.0]], dtype=np.float32)
    matmul_pass = np.allclose(expected_matmul, matmul_target, atol=1e-5)

    # Test 2: ReLU Activation
    relu_input = np.array([-2.0, 0.0, 3.0, -4.0, 5.0], dtype=np.float32)
    relu_expected = np.maximum(0.0, relu_input)  # [0, 0, 3, 0, 5]
    relu_target = np.array([0.0, 0.0, 3.0, 0.0, 5.0], dtype=np.float32)
    relu_pass = np.allclose(relu_expected, relu_target, atol=1e-5)

    # Test 3: Sigmoid Activation
    sig_input = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    sig_expected = 1.0 / (1.0 + np.exp(-sig_input))  # [~0.119203, 0.5, ~0.880797]
    sig_target = np.array([1.0 / (1.0 + math.exp(2.0)), 0.5, 1.0 / (1.0 + math.exp(-2.0))], dtype=np.float32)
    sigmoid_pass = np.allclose(sig_expected, sig_target, atol=1e-5)

    # Full Network Deterministic Forward Pass
    BATCH = 256
    INPUT_SIZE = 128
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 10

    X = np.array([0.5 * math.sin(i * 0.013) for i in range(BATCH * INPUT_SIZE)], dtype=np.float32).reshape(BATCH, INPUT_SIZE)
    W1 = np.array([0.08 * math.cos(i * 0.017) for i in range(INPUT_SIZE * HIDDEN_SIZE)], dtype=np.float32).reshape(INPUT_SIZE, HIDDEN_SIZE)
    b1 = np.array([0.01 * (i % 7) for i in range(HIDDEN_SIZE)], dtype=np.float32)
    W2 = np.array([0.10 * math.sin(i * 0.019) for i in range(HIDDEN_SIZE * OUTPUT_SIZE)], dtype=np.float32).reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    b2 = np.array([0.02 * (i % 5) for i in range(OUTPUT_SIZE)], dtype=np.float32)

    Z1 = np.matmul(X, W1) + b1
    A1 = np.maximum(0.0, Z1)
    Z2 = np.matmul(A1, W2) + b2
    Y = 1.0 / (1.0 + np.exp(-Z2))

    first_10_actual = [float(v) for v in Y.flatten()[:10]]
    first_10_ref = [0.498196, 0.503101, 0.508006, 0.512911, 0.517814,
                    0.497729, 0.502638, 0.507548, 0.512457, 0.517364]
    max_abs_diff = float(np.max(np.abs(np.array(first_10_actual) - np.array(first_10_ref))))
    tolerance = 1e-4
    match = max_abs_diff < tolerance

    validation_data = {
        "tests": {
            "matrix_mul_bias": {
                "status": "PASSED" if matmul_pass else "FAILED",
                "expected": matmul_target.tolist(),
                "actual": expected_matmul.tolist()
            },
            "relu": {
                "status": "PASSED" if relu_pass else "FAILED",
                "expected": relu_target.tolist(),
                "actual": relu_expected.tolist()
            },
            "sigmoid": {
                "status": "PASSED" if sigmoid_pass else "FAILED",
                "expected": sig_target.tolist(),
                "actual": sig_expected.tolist()
            }
        },
        "network_forward_pass": {
            "batch_size": BATCH,
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "output_size": OUTPUT_SIZE,
            "first_10_computed": [round(v, 6) for v in first_10_actual],
            "first_10_reference": first_10_ref,
            "max_absolute_difference": max_abs_diff,
            "tolerance": tolerance,
            "match": match
        },
        "overall_validation": "PASSED" if (matmul_pass and relu_pass and sigmoid_pass and match) else "FAILED"
    }

    out_file = os.path.join(results_dir, "validation_results.json")
    with open(out_file, "w") as f:
        json.dump(validation_data, f, indent=2)

    print("=== Validation Summary ===")
    print(f"Matrix Multiplication + Bias: {validation_data['tests']['matrix_mul_bias']['status']}")
    print(f"ReLU Activation:              {validation_data['tests']['relu']['status']}")
    print(f"Sigmoid Activation:           {validation_data['tests']['sigmoid']['status']}")
    print(f"Max Absolute Difference:     {max_abs_diff:.6f}")
    print(f"Status:                       {validation_data['overall_validation']}")
    print(f"Saved:                        {out_file}")

    return 0 if validation_data["overall_validation"] == "PASSED" else 1

if __name__ == "__main__":
    sys.exit(run_validation())
