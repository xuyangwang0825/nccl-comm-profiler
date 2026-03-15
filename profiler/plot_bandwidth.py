#!/usr/bin/env python3
"""
plot_bandwidth.py

Generates bandwidth-vs-message-size plots from nccl_bench and ring_allreduce
benchmark output. Produces the kind of charts you'd include in a README or
present in an interview to explain performance analysis.

Plots generated:
  1. busbw vs message size for all 4 NCCL collectives
  2. NCCL AllReduce vs custom ring_allreduce (the key comparison)
  3. Latency (us) vs message size — useful for small-message analysis

Usage:
    # After running benchmarks:
    ./build/nccl_bench    > results/nccl_bench.csv
    ./build/ring_allreduce > results/ring_allreduce.csv

    python3 profiler/plot_bandwidth.py \
        --nccl   results/nccl_bench.csv \
        --ring   results/ring_allreduce.csv \
        --outdir results/plots
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

# matplotlib is optional — skip plotting if not installed, still print tables
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend (works on headless GPU servers)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not installed. Install with: pip install matplotlib")
    print("       Tables will be printed but no plots saved.\n")


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_nccl_csv(path: str) -> dict[str, list[dict]]:
    """Load nccl_bench CSV. Returns {op_name: [row, ...]}."""
    data: dict[str, list] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            # Handle both raw nccl_bench output and parsed CSV
            try:
                if parts[0].replace('.', '').isdigit():
                    # parsed CSV: size_bytes, time_us, algbw, busbw (no op column)
                    op = 'AllReduce'
                    row = {'op': op, 'size_bytes': int(parts[0]),
                           'time_us': float(parts[1]),
                           'algbw_GBps': float(parts[2]),
                           'busbw_GBps': float(parts[3])}
                else:
                    # raw nccl_bench output: op, size_bytes, count, time_us, algbw, busbw
                    op = parts[0]
                    row = {'op': op, 'size_bytes': int(parts[1]),
                           'time_us': float(parts[3]),
                           'algbw_GBps': float(parts[4]),
                           'busbw_GBps': float(parts[5])}
                data.setdefault(op, []).append(row)
            except (ValueError, IndexError):
                continue
    return data


def load_ring_csv(path: str) -> list[dict]:
    """Load ring_allreduce CSV output (size_bytes,time_us,algbw_GBps,busbw_GBps)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
            try:
                rows.append({
                    'size_bytes': int(parts[0]),
                    'time_us':    float(parts[1]),
                    'algbw_GBps': float(parts[2]),
                    'busbw_GBps': float(parts[3]),
                })
            except ValueError:
                continue
    return rows


# ─── Plotting helpers ─────────────────────────────────────────────────────────

COLORS = {
    'AllReduce':     '#e74c3c',
    'AllGather':     '#3498db',
    'ReduceScatter': '#2ecc71',
    'Broadcast':     '#9b59b6',
    'ring_custom':   '#f39c12',
}

OP_LABELS = {
    'AllReduce':     'AllReduce (NCCL)',
    'AllGather':     'AllGather (NCCL)',
    'ReduceScatter': 'ReduceScatter (NCCL)',
    'Broadcast':     'Broadcast (NCCL)',
    'ring_custom':   'Ring AllReduce (custom)',
}

def fmt_bytes(n: float) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def setup_size_xaxis(ax):
    """Log2 x-axis labeled in human-readable byte sizes."""
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: fmt_bytes(x))
    )
    ax.tick_params(axis='x', rotation=30)


# ─── Plot 1: All four NCCL collectives, busbw vs size ────────────────────────

def plot_all_collectives(nccl_data: dict, outdir: str) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 6))

    for op, rows in nccl_data.items():
        if not rows:
            continue
        xs = [r['size_bytes'] for r in rows]
        ys = [r['busbw_GBps'] for r in rows]
        ax.plot(xs, ys, marker='o', markersize=4,
                color=COLORS.get(op, 'gray'),
                label=OP_LABELS.get(op, op))

    setup_size_xaxis(ax)
    ax.set_xlabel('Message Size')
    ax.set_ylabel('Bus Bandwidth (GB/s)')
    ax.set_title('NCCL Collective Operations — Bus Bandwidth vs Message Size')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(bottom=0)

    out = Path(outdir) / 'nccl_all_collectives.png'
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ─── Plot 2: NCCL AllReduce vs custom ring-allreduce ─────────────────────────

def plot_nccl_vs_ring(nccl_data: dict, ring_rows: list, outdir: str) -> None:
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    nccl_ar = nccl_data.get('AllReduce', [])

    # Left: bus bandwidth
    ax = axes[0]
    if nccl_ar:
        ax.plot([r['size_bytes'] for r in nccl_ar],
                [r['busbw_GBps']  for r in nccl_ar],
                marker='o', markersize=4, color=COLORS['AllReduce'],
                label='NCCL AllReduce')
    if ring_rows:
        ax.plot([r['size_bytes'] for r in ring_rows],
                [r['busbw_GBps']  for r in ring_rows],
                marker='s', markersize=4, color=COLORS['ring_custom'],
                linestyle='--', label='Custom Ring (ours)')

    setup_size_xaxis(ax)
    ax.set_xlabel('Message Size')
    ax.set_ylabel('Bus Bandwidth (GB/s)')
    ax.set_title('AllReduce: NCCL vs Custom Ring\n(Bus Bandwidth)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(bottom=0)

    # Right: latency
    ax = axes[1]
    if nccl_ar:
        ax.plot([r['size_bytes'] for r in nccl_ar],
                [r['time_us']    for r in nccl_ar],
                marker='o', markersize=4, color=COLORS['AllReduce'],
                label='NCCL AllReduce')
    if ring_rows:
        ax.plot([r['size_bytes'] for r in ring_rows],
                [r['time_us']    for r in ring_rows],
                marker='s', markersize=4, color=COLORS['ring_custom'],
                linestyle='--', label='Custom Ring (ours)')

    setup_size_xaxis(ax)
    ax.set_xlabel('Message Size')
    ax.set_ylabel('Latency (µs)')
    ax.set_title('AllReduce: NCCL vs Custom Ring\n(Latency)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    out = Path(outdir) / 'nccl_vs_ring_allreduce.png'
    fig.suptitle('NCCL vs Hand-rolled Ring AllReduce', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ─── Plot 3: Alpha-beta model fit ─────────────────────────────────────────────
#
# The α-β model: latency(size) = α + β * size
#   α = startup/injection latency (microseconds)
#   β = 1 / bandwidth (us/byte)
#
# Fitting this to measured data lets you estimate peak bandwidth and
# compare against theoretical link speed. This is what the resume
# bullet about the "α-β model" refers to.

def plot_alpha_beta(rows: list, label: str, outdir: str) -> None:
    if not HAS_MPL or not rows:
        return
    try:
        import numpy as np
    except ImportError:
        print("[warn] numpy not installed, skipping alpha-beta plot")
        return

    sizes  = np.array([r['size_bytes'] for r in rows], dtype=float)
    latencies = np.array([r['time_us'] for r in rows], dtype=float)

    # Linear fit: latency = alpha + beta * size
    coeffs = np.polyfit(sizes, latencies, 1)
    beta, alpha = coeffs  # slope=beta, intercept=alpha
    peak_bw_GBps = 1e6 / (beta * 1e9)  # beta in us/byte → GB/s

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(sizes, latencies, s=20, label='Measured', zorder=3)
    fit_y = alpha + beta * sizes
    ax.plot(sizes, fit_y, 'r--',
            label=f'α-β fit  α={alpha:.1f}µs  β={beta*1e9:.2f}ns/B  peak={peak_bw_GBps:.1f} GB/s')
    setup_size_xaxis(ax)
    ax.set_xlabel('Message Size')
    ax.set_ylabel('Latency (µs)')
    ax.set_title(f'{label}: α-β Latency Model')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    out = Path(outdir) / f'alpha_beta_{label.replace(" ", "_")}.png'
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}  (α={alpha:.1f}µs, peak={peak_bw_GBps:.1f} GB/s)")


# ─── Text table fallback ───────────────────────────────────────────────────────

def print_comparison_table(nccl_data: dict, ring_rows: list) -> None:
    nccl_ar = {r['size_bytes']: r for r in nccl_data.get('AllReduce', [])}
    ring_map = {r['size_bytes']: r for r in ring_rows}
    common   = sorted(set(nccl_ar) & set(ring_map))

    if not common:
        print("No overlapping sizes to compare.")
        return

    print(f"\n{'Size':>12}  {'NCCL busbw':>12}  {'Ring busbw':>12}  {'Speedup':>10}")
    print("-" * 52)
    for sz in common:
        n = nccl_ar[sz]['busbw_GBps']
        r = ring_map[sz]['busbw_GBps']
        speedup = n / r if r > 0 else float('inf')
        print(f"{fmt_bytes(sz):>12}  {n:>10.2f}  {r:>10.2f}  {speedup:>9.2f}x")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nccl',   metavar='FILE', help='nccl_bench CSV output')
    parser.add_argument('--ring',   metavar='FILE', help='ring_allreduce CSV output')
    parser.add_argument('--outdir', metavar='DIR',  default='results/plots',
                        help='Output directory for plots (default: results/plots)')
    args = parser.parse_args()

    if not args.nccl and not args.ring:
        parser.print_help()
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    nccl_data: dict[str, list] = {}
    ring_rows: list = []

    if args.nccl:
        nccl_data = load_nccl_csv(args.nccl)
        print(f"Loaded NCCL data: {sum(len(v) for v in nccl_data.values())} rows, "
              f"ops: {list(nccl_data.keys())}")
        plot_all_collectives(nccl_data, args.outdir)
        if 'AllReduce' in nccl_data:
            plot_alpha_beta(nccl_data['AllReduce'], 'NCCL_AllReduce', args.outdir)

    if args.ring:
        ring_rows = load_ring_csv(args.ring)
        print(f"Loaded ring data: {len(ring_rows)} rows")
        plot_alpha_beta(ring_rows, 'Custom_Ring', args.outdir)

    if args.nccl and args.ring:
        plot_nccl_vs_ring(nccl_data, ring_rows, args.outdir)
        print_comparison_table(nccl_data, ring_rows)

    if not HAS_MPL:
        print("\nInstall matplotlib to generate plots: pip install matplotlib numpy")


if __name__ == '__main__':
    main()
