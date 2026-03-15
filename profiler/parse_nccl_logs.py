#!/usr/bin/env python3
"""
parse_nccl_logs.py

Parses two kinds of input:

1. NCCL_DEBUG=INFO stderr logs → extracts topology info (NVLink vs IB,
   algorithm chosen, ring order) as a JSON summary.

2. nccl_bench CSV stdout → structured DataFrame for plotting.

Usage:
    # Parse NCCL_DEBUG logs
    NCCL_DEBUG=INFO ./build/nccl_bench 2>nccl_debug.log
    python3 profiler/parse_nccl_logs.py --log nccl_debug.log

    # Parse benchmark CSV output
    ./build/nccl_bench > bench.csv
    python3 profiler/parse_nccl_logs.py --csv bench.csv

    # Combine: parse both and annotate the CSV with topology
    python3 profiler/parse_nccl_logs.py --log nccl_debug.log --csv bench.csv
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class RankInfo:
    rank: int
    device: str = ""            # e.g. "mlx5_0:1/IB"
    transport: str = ""         # "NVLink" | "InfiniBand" | "NET" | "SHM"
    algo: str = ""              # "Ring" | "Tree" | "CollNet"
    ring_order: list = field(default_factory=list)

@dataclass
class TopologySummary:
    n_ranks: int = 0
    nvlink_pairs: list = field(default_factory=list)   # (g1, g2) pairs with NVLink
    ib_device: str = ""         # InfiniBand HCA in use
    algorithm: str = ""         # NCCL-chosen algorithm
    ring_orders: list = field(default_factory=list)    # one ring per channel
    notes: list = field(default_factory=list)

# ─── Log parsing ──────────────────────────────────────────────────────────────

# NCCL_DEBUG=INFO log lines look like:
#   hostname:pid:tid [rank] NCCL INFO <message>

RE_PREFIX  = re.compile(r'\[(\d+)\] NCCL INFO (.+)')
RE_NVLINK  = re.compile(r'GPU (\d+) GPU (\d+) via NVLink')
RE_IB      = re.compile(r'NET/IB\s*:.*?(\S+:\d+/\S+)')
RE_RING    = re.compile(r'Ring\s+\d+\s*:\s*([\d\s]+)')
RE_ALGO    = re.compile(r'AllReduce.*?algo\s+(\w+)', re.IGNORECASE)
RE_NRANKS  = re.compile(r'\[nranks=(\d+)\]')
RE_ALGO2   = re.compile(r'Using\s+(Ring|Tree|CollNet)\s+algorithm', re.IGNORECASE)
RE_SLOW    = re.compile(r'Timeout.*rank\s+(\d+)', re.IGNORECASE)


def parse_debug_log(log_text: str) -> TopologySummary:
    """Extract topology metadata from NCCL_DEBUG=INFO stderr output."""
    topo = TopologySummary()

    for line in log_text.splitlines():
        m = RE_PREFIX.search(line)
        if not m:
            continue
        rank_str, msg = m.group(1), m.group(2)

        # Number of ranks
        nr = RE_NRANKS.search(msg)
        if nr and topo.n_ranks == 0:
            topo.n_ranks = int(nr.group(1))

        # NVLink pairs detected by NCCL topology detection
        for nv in RE_NVLINK.finditer(msg):
            pair = (int(nv.group(1)), int(nv.group(2)))
            if pair not in topo.nvlink_pairs:
                topo.nvlink_pairs.append(pair)

        # InfiniBand HCA
        ib = RE_IB.search(msg)
        if ib and not topo.ib_device:
            topo.ib_device = ib.group(1)

        # Ring order (one line per channel, e.g. "Ring 00 : 0 1 3 2")
        rng = RE_RING.search(msg)
        if rng:
            order = list(map(int, rng.group(1).split()))
            if order not in topo.ring_orders:
                topo.ring_orders.append(order)

        # Algorithm selection
        a2 = RE_ALGO2.search(msg)
        if a2 and not topo.algorithm:
            topo.algorithm = a2.group(1)

        # Slow rank detection (if any timeouts appear)
        sl = RE_SLOW.search(msg)
        if sl:
            topo.notes.append(f"Slow rank detected: rank {sl.group(1)}")

    # Infer interconnect type from what we found
    if topo.nvlink_pairs:
        n = topo.n_ranks or "?"
        topo.notes.append(
            f"NVLink detected between {len(topo.nvlink_pairs)} GPU pairs — "
            f"expect high busbw (~300 GB/s on A100)"
        )
    elif topo.ib_device:
        topo.notes.append(
            f"InfiniBand transport ({topo.ib_device}) — "
            f"cross-node communication via RDMA"
        )
    else:
        topo.notes.append(
            "No NVLink or IB detected — GPUs communicate via PCIe (~32 GB/s busbw)"
        )

    return topo


# ─── CSV parsing ──────────────────────────────────────────────────────────────

def parse_bench_csv(csv_text: str) -> list[dict]:
    """
    Parse CSV output from nccl_bench or nccl-tests.

    nccl_bench format:
        op,size_bytes,count,time_us,algbw_GBps,busbw_GBps

    nccl-tests format (space-aligned):
        #       size         count  type  redop  root  time   algbw  busbw  #wrong
        <values>
    """
    rows = []
    nccl_tests_mode = False

    for line in csv_text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            # nccl-tests header detection
            if 'algbw' in line and 'busbw' in line and 'size' in line:
                nccl_tests_mode = True
            continue

        if nccl_tests_mode:
            # nccl-tests outputs space-separated columns
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                rows.append({
                    'op': 'AllReduce',   # nccl-tests runs one op at a time
                    'size_bytes': int(parts[0]),
                    'count': int(parts[1]),
                    'time_us': float(parts[5]) * 1000,   # nccl-tests reports ms
                    'algbw_GBps': float(parts[6]),
                    'busbw_GBps': float(parts[7]),
                })
            except (ValueError, IndexError):
                continue
        else:
            # Our nccl_bench CSV format
            parts = line.split(',')
            if len(parts) < 6:
                continue
            try:
                rows.append({
                    'op': parts[0],
                    'size_bytes': int(parts[1]),
                    'count': int(parts[2]),
                    'time_us': float(parts[3]),
                    'algbw_GBps': float(parts[4]),
                    'busbw_GBps': float(parts[5]),
                })
            except (ValueError, IndexError):
                continue
    return rows


# ─── Slow-rank detection ───────────────────────────────────────────────────────

def detect_slow_ranks(per_rank_times: dict[int, list[float]],
                      threshold_pct: float = 20.0) -> list[int]:
    """
    Given per-rank timing data {rank: [t1, t2, ...]}, flag ranks that are
    consistently slower than the mean by more than `threshold_pct` percent.

    In real NCCL collectives, a single slow rank stalls all others because
    they can't complete AllReduce until every rank has contributed its data.
    Identifying the slow rank is a key debugging skill.
    """
    if not per_rank_times:
        return []

    mean_times = {r: sum(ts) / len(ts) for r, ts in per_rank_times.items()}
    overall_mean = sum(mean_times.values()) / len(mean_times)

    slow = [r for r, t in mean_times.items()
            if t > overall_mean * (1 + threshold_pct / 100)]
    return slow


# ─── Report ────────────────────────────────────────────────────────────────────

def print_topology_report(topo: TopologySummary) -> None:
    print("=" * 60)
    print("  NCCL Topology Report")
    print("=" * 60)
    print(f"  Ranks       : {topo.n_ranks}")
    print(f"  Algorithm   : {topo.algorithm or 'not detected'}")
    print(f"  IB device   : {topo.ib_device or 'none'}")
    if topo.nvlink_pairs:
        pairs = ', '.join(f"({a},{b})" for a,b in topo.nvlink_pairs)
        print(f"  NVLink pairs: {pairs}")
    else:
        print(f"  NVLink pairs: none (PCIe only)")
    if topo.ring_orders:
        for i, order in enumerate(topo.ring_orders):
            print(f"  Ring ch {i}   : {' → '.join(map(str, order))}")
    print()
    for note in topo.notes:
        print(f"  ⚡ {note}")
    print("=" * 60)


def print_csv_summary(rows: list[dict]) -> None:
    if not rows:
        print("No CSV rows parsed.")
        return

    ops = sorted(set(r['op'] for r in rows))
    print(f"\n{'Op':<16} {'Size':>12} {'time(us)':>12} {'algbw':>10} {'busbw':>10}")
    print("-" * 64)
    for op in ops:
        op_rows = [r for r in rows if r['op'] == op]
        # Show min, mid, max sizes
        for r in [op_rows[0], op_rows[len(op_rows)//2], op_rows[-1]]:
            sz_str = f"{r['size_bytes'] / 1e6:.1f} MB"
            print(f"{r['op']:<16} {sz_str:>12} {r['time_us']:>12.1f} "
                  f"{r['algbw_GBps']:>10.2f} {r['busbw_GBps']:>10.2f}")
        print()


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--log', metavar='FILE',
                        help='NCCL_DEBUG=INFO log file (stderr output)')
    parser.add_argument('--csv', metavar='FILE',
                        help='nccl_bench or nccl-tests CSV output file')
    parser.add_argument('--json-out', metavar='FILE',
                        help='Write topology summary as JSON to this file')
    args = parser.parse_args()

    if not args.log and not args.csv:
        parser.print_help()
        sys.exit(1)

    # ── Parse topology from debug log ──────────────────────────────────────
    topo: Optional[TopologySummary] = None
    if args.log:
        log_text = Path(args.log).read_text()
        topo = parse_debug_log(log_text)
        print_topology_report(topo)

        if args.json_out:
            import dataclasses
            with open(args.json_out, 'w') as f:
                json.dump(dataclasses.asdict(topo), f, indent=2)
            print(f"\nTopology JSON written to {args.json_out}")

    # ── Parse benchmark CSV ────────────────────────────────────────────────
    rows: list[dict] = []
    if args.csv:
        csv_text = Path(args.csv).read_text()
        rows = parse_bench_csv(csv_text)
        print_csv_summary(rows)

        # Write structured CSV for the plotter
        out_path = Path(args.csv).with_suffix('.parsed.csv')
        with open(out_path, 'w') as f:
            f.write("op,size_bytes,time_us,algbw_GBps,busbw_GBps\n")
            for r in rows:
                f.write(f"{r['op']},{r['size_bytes']},{r['time_us']:.2f},"
                        f"{r['algbw_GBps']:.4f},{r['busbw_GBps']:.4f}\n")
        print(f"Parsed CSV written to {out_path}")

    # ── Annotate: warn if NVLink absent but we expect it ──────────────────
    if topo and rows:
        max_busbw = max(r['busbw_GBps'] for r in rows)
        if not topo.nvlink_pairs and max_busbw < 50:
            print("\n[NOTE] Peak busbw < 50 GB/s and no NVLink detected.")
            print("       This is expected for PCIe-only setups.")
            print("       On NVLink (A100): expect ~300 GB/s.")
        elif topo.nvlink_pairs and max_busbw > 100:
            print(f"\n[OK] Peak busbw = {max_busbw:.1f} GB/s — NVLink is active.")


if __name__ == '__main__':
    main()
