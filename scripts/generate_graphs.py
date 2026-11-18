#!/usr/bin/env python3
"""
Generate publication-quality graphs for Experiment 5.
Visualizes performance comparisons, speedups, and timing variations.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    graphs_dir = os.path.join(results_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "benchmark_results.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract metrics
    observed_cpu_ms = data["measured_summary"]["average_cpu_ms"]
    ref_cpu_ms = data["reference_benchmark"]["cpu_ms"]
    ref_gpu_kernel_ms = data["reference_benchmark"]["gpu_kernel_only_ms"]
    ref_gpu_e2e_ms = data["reference_benchmark"]["gpu_end_to_end_ms"]
    ref_kernel_speedup = data["reference_benchmark"]["kernel_only_speedup"]
    ref_e2e_speedup = data["reference_benchmark"]["end_to_end_speedup"]
    measured_runs = data["measured_runs"]

    # Graph theme settings
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.sans-serif": "Arial, Helvetica, DejaVu Sans",
        "font.family": "sans-serif",
        "figure.titlesize": 14,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.autolayout": True
    })

    # Color palette
    c_blue = "#1f77b4"
    c_teal = "#17becf"
    c_orange = "#ff7f0e"
    c_green = "#2ca02c"
    c_purple = "#9467bd"

    # -------------------------------------------------------------
    # Graph 1: CPU vs GPU Kernel-Only Execution Time
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    labels = ["Observed Host CPU\n(i7/Ryzen)", "Ref CPU\n(Lab Workstation)", "Ref GPU Kernel\n(RTX 4500 Ada)"]
    times = [observed_cpu_ms, ref_cpu_ms, ref_gpu_kernel_ms]
    colors = [c_blue, "#4b8bbe", c_green]

    bars = ax.bar(labels, times, color=colors, width=0.45, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Execution Time (milliseconds, ms)")
    ax.set_title("Experiment 5: CPU vs GPU Kernel-Only Execution Time\n(Batch 256, Hidden 64, 20 Repeats)", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.015, f"{h:.4f} ms",
                ha="center", va="bottom", fontweight="bold", fontsize=9)

    ax.set_ylim(0, max(times) * 1.18)
    fig.savefig(os.path.join(graphs_dir, "01_cpu_vs_gpu_kernel_only.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Graph 2: CPU vs GPU End-to-End Execution Time
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    labels = ["Observed Host CPU\n(Direct Compute)", "Ref CPU\n(Direct Compute)", "Ref GPU End-to-End\n(PCIe Transfers + Kernels)"]
    times = [observed_cpu_ms, ref_cpu_ms, ref_gpu_e2e_ms]
    colors = [c_blue, "#4b8bbe", c_orange]

    bars = ax.bar(labels, times, color=colors, width=0.45, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Execution Time (milliseconds, ms)")
    ax.set_title("Experiment 5: CPU vs GPU End-to-End Execution Time\n(Including Host-to-Device and Device-to-Host Copies)", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.015, f"{h:.4f} ms",
                ha="center", va="bottom", fontweight="bold", fontsize=9)

    ax.set_ylim(0, max(times) * 1.18)
    fig.savefig(os.path.join(graphs_dir, "02_cpu_vs_gpu_end_to_end.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Graph 3: Kernel-Only vs End-to-End GPU Execution Time
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    gpu_labels = ["GPU Kernel-Only\n(Compute in VRAM)", "GPU End-to-End\n(Compute + H2D/D2H Transfers)"]
    gpu_times = [ref_gpu_kernel_ms, ref_gpu_e2e_ms]
    gpu_colors = [c_green, c_orange]

    bars = ax.bar(gpu_labels, gpu_times, color=gpu_colors, width=0.4, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Time (milliseconds, ms)")
    ax.set_title("Experiment 5: GPU Kernel-Only vs End-to-End Time\nMemory Transfer Overhead Breakdown", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.002, f"{h:.4f} ms",
                ha="center", va="bottom", fontweight="bold", fontsize=9)

    transfer_overhead = ref_gpu_e2e_ms - ref_gpu_kernel_ms
    pct_transfer = (transfer_overhead / ref_gpu_e2e_ms) * 100.0
    ax.text(0.5, max(gpu_times) * 0.75, f"Transfer Overhead: {transfer_overhead:.4f} ms ({pct_transfer:.1f}%)",
            ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2cc", edgecolor="#d6b656"))

    ax.set_ylim(0, max(gpu_times) * 1.25)
    fig.savefig(os.path.join(graphs_dir, "03_gpu_kernel_vs_end_to_end.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Graph 4: Kernel-Only vs End-to-End Speedup
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    speedup_labels = ["Kernel-Only Speedup\n(Compute Scaling)", "End-to-End Speedup\n(Wall-Clock System Scaling)"]
    speedups = [ref_kernel_speedup, ref_e2e_speedup]
    s_colors = [c_green, c_teal]

    bars = ax.bar(speedup_labels, speedups, color=s_colors, width=0.4, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Speedup Multiplier (x)")
    ax.set_title("Experiment 5: GPU Speedup Over CPU Forward Pass\n(Reference NVIDIA RTX 4500 Ada Generation)", pad=15)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="CPU Baseline (1.0x)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, f"{h:.2f}x",
                ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylim(0, max(speedups) * 1.2)
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(graphs_dir, "04_speedup_comparison.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Graph 5: Benchmark Timing Variation Across Runs
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    run_indices = [f"Run {r['run_index']}" for r in measured_runs]
    cpu_run_times = [r["cpu_time_ms"] for r in measured_runs]

    ax.plot(run_indices, cpu_run_times, marker="o", color=c_blue, linewidth=2, markersize=8, label="Observed CPU Time (ms)")
    ax.axhline(observed_cpu_ms, color="red", linestyle=":", linewidth=1.5, label=f"Observed Mean ({observed_cpu_ms:.4f} ms)")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title("Experiment 5: Timing Stability Across Successive Benchmark Runs\n(20 Repetitions per Benchmark Run)", pad=15)
    ax.grid(True, linestyle="--", alpha=0.6)

    for i, t in enumerate(cpu_run_times):
        ax.text(i, t + 0.001, f"{t:.4f} ms", ha="center", va="bottom", fontweight="bold", fontsize=9)

    margin = (max(cpu_run_times) - min(cpu_run_times)) * 1.5 if max(cpu_run_times) != min(cpu_run_times) else 0.05
    ax.set_ylim(min(cpu_run_times) - margin, max(cpu_run_times) + margin)
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(graphs_dir, "05_run_stability_variation.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Combined Composite Figure: 00_benchmark_overview.png
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)

    # Subplot 1: Times
    axes[0, 0].bar(["Obs CPU", "Ref CPU", "GPU Kernel", "GPU E2E"],
                   [observed_cpu_ms, ref_cpu_ms, ref_gpu_kernel_ms, ref_gpu_e2e_ms],
                   color=[c_blue, "#4b8bbe", c_green, c_orange], edgecolor="black")
    axes[0, 0].set_ylabel("Time (ms)")
    axes[0, 0].set_title("Execution Times by Platform and Scope")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.6)

    # Subplot 2: Speedups
    axes[0, 1].bar(["Kernel-Only", "End-to-End"], [ref_kernel_speedup, ref_e2e_speedup],
                   color=[c_green, c_teal], edgecolor="black")
    axes[0, 1].set_ylabel("Speedup (x)")
    axes[0, 1].set_title("GPU Acceleration Multiplier vs CPU")
    axes[0, 1].axhline(1.0, color="gray", linestyle="--")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.6)

    # Subplot 3: Run Variation
    axes[1, 0].plot(run_indices, cpu_run_times, marker="s", color=c_blue, linewidth=2)
    axes[1, 0].axhline(observed_cpu_ms, color="red", linestyle=":")
    axes[1, 0].set_ylabel("CPU Time (ms)")
    axes[1, 0].set_title("Run Stability Variation")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # Subplot 4: Transfer Overhead
    axes[1, 1].pie([ref_gpu_kernel_ms, transfer_overhead],
                   labels=[f"Kernel Compute\n({ref_gpu_kernel_ms:.3f} ms)", f"PCIe Transfers\n({transfer_overhead:.3f} ms)"],
                   autopct="%1.1f%%", colors=[c_green, c_orange], startangle=140,
                   wedgeprops=dict(edgecolor="black", linewidth=1))
    axes[1, 1].set_title("GPU End-to-End Latency Composition")

    fig.suptitle("Experiment 5: Neural Network Layers in CUDA — Performance Overview", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "00_benchmark_overview.png"))
    plt.close(fig)

    print("Generated 6 high-resolution performance graphs successfully:")
    for f in os.listdir(graphs_dir):
        print(f"  {f}")

if __name__ == "__main__":
    main()
