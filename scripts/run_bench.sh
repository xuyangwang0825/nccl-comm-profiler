#!/usr/bin/env bash
# run_bench.sh
#
# One-script setup and benchmark runner for nccl-comm-profiler.
# Run this on your GPU instance after building the project.
#
# Usage:
#   chmod +x scripts/run_bench.sh
#   ./scripts/run_bench.sh [n_gpus]          # default: auto-detect
#
# What this does:
#   1. Detect available GPUs
#   2. Print GPU topology (nvidia-smi topo -m)
#   3. Run nccl_bench → save CSV
#   4. Run ring_allreduce → save CSV
#   5. Capture NCCL_DEBUG log from nccl_bench
#   6. Parse logs and generate plots

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
BUILD_DIR="$(pwd)/build"
RESULTS_DIR="$(pwd)/results"
PLOTS_DIR="${RESULTS_DIR}/plots"

# Auto-detect GPU count if not passed
if [[ $# -ge 1 ]]; then
    N_GPUS="$1"
else
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
fi

echo "=================================================="
echo "  NCCL Comm Profiler — Benchmark Runner"
echo "  GPUs: ${N_GPUS}"
echo "=================================================="

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -f "${BUILD_DIR}/nccl_bench" ]]; then
    echo "[error] Build artifacts not found. Run:"
    echo "    mkdir build && cd build && cmake .. && make -j4"
    exit 1
fi

mkdir -p "${RESULTS_DIR}" "${PLOTS_DIR}"

# ── Step 1: GPU topology ───────────────────────────────────────────────────────
echo ""
echo "── GPU Topology ──────────────────────────────────"
nvidia-smi topo -m 2>/dev/null || echo "[warn] nvidia-smi topo not available"

echo ""
echo "── GPU Info ──────────────────────────────────────"
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current \
           --format=csv 2>/dev/null || true

# ── Step 2: Run nccl_bench (capture stdout=CSV, stderr=NCCL_DEBUG) ────────────
echo ""
echo "── Running NCCL Benchmark ────────────────────────"
echo "   (this takes ~2 minutes for all ops × all sizes)"

NCCL_DEBUG=INFO \
    "${BUILD_DIR}/nccl_bench" "${N_GPUS}" \
    > "${RESULTS_DIR}/nccl_bench.csv" \
    2> "${RESULTS_DIR}/nccl_debug.log"

echo "   Saved CSV  : ${RESULTS_DIR}/nccl_bench.csv"
echo "   Saved log  : ${RESULTS_DIR}/nccl_debug.log"

# Quick sanity: show peak AllReduce busbw
PEAK_BUSBW=$(grep '^AllReduce' "${RESULTS_DIR}/nccl_bench.csv" \
             | awk -F',' '{print $NF}' \
             | sort -n \
             | tail -1)
echo "   Peak AllReduce busbw: ${PEAK_BUSBW} GB/s"

# ── Step 3: Run ring_allreduce ─────────────────────────────────────────────────
echo ""
echo "── Running Custom Ring AllReduce ─────────────────"

"${BUILD_DIR}/ring_allreduce" "${N_GPUS}" \
    > "${RESULTS_DIR}/ring_allreduce.csv" \
    2>&1

echo "   Saved CSV  : ${RESULTS_DIR}/ring_allreduce.csv"

# ── Step 4: (Optional) Run official nccl-tests for cross-validation ───────────
if command -v ./nccl-tests/build/all_reduce_perf &>/dev/null; then
    echo ""
    echo "── nccl-tests AllReduce (cross-validation) ───────"
    ./nccl-tests/build/all_reduce_perf \
        -b 8 -e 256M -f 2 -g "${N_GPUS}" \
        > "${RESULTS_DIR}/nccl_tests_allreduce.txt" 2>&1
    echo "   Saved: ${RESULTS_DIR}/nccl_tests_allreduce.txt"
fi

# ── Step 5: Parse logs ────────────────────────────────────────────────────────
echo ""
echo "── Parsing NCCL Debug Logs ───────────────────────"
python3 profiler/parse_nccl_logs.py \
    --log "${RESULTS_DIR}/nccl_debug.log" \
    --csv "${RESULTS_DIR}/nccl_bench.csv" \
    --json-out "${RESULTS_DIR}/topology.json" \
    2>/dev/null || echo "[warn] parse_nccl_logs.py failed (check Python deps)"

# ── Step 6: Generate plots ────────────────────────────────────────────────────
echo ""
echo "── Generating Plots ──────────────────────────────"
python3 profiler/plot_bandwidth.py \
    --nccl   "${RESULTS_DIR}/nccl_bench.csv" \
    --ring   "${RESULTS_DIR}/ring_allreduce.csv" \
    --outdir "${PLOTS_DIR}" \
    2>/dev/null || echo "[warn] plot_bandwidth.py failed (pip install matplotlib numpy)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Results saved to: ${RESULTS_DIR}/"
echo "  Plots   saved to: ${PLOTS_DIR}/"
echo ""
echo "  Key files:"
echo "    nccl_bench.csv         — NCCL collective benchmarks"
echo "    ring_allreduce.csv     — Custom ring implementation"
echo "    nccl_debug.log         — NCCL_DEBUG=INFO output (topology)"
echo "    topology.json          — Parsed topology summary"
echo "    plots/                 — Bandwidth graphs"
echo "=================================================="
