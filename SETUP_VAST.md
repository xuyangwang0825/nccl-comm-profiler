# Vast.ai Setup Guide

How to spin up a 2-GPU instance on Vast.ai and run this profiler end-to-end.
Good fallback when Lambda Cloud is out of capacity — usually cheaper and more available.

> **Prefer Lambda?** See [SETUP_LAMBDA.md](SETUP_LAMBDA.md).

---

## 1. Create Account & Add SSH Key

1. Go to [vast.ai](https://vast.ai) → Sign up
2. **Account → SSH Keys → Add SSH Key**
   ```bash
   # On your Mac, generate a key if you don't have one
   ssh-keygen -t ed25519 -C "vast"
   cat ~/.ssh/id_ed25519.pub   # copy this output into Vast's SSH key field
   ```

---

## 2. Find a 2-GPU Instance

**Search → Rent** — use these filters:

```
GPU Count:  2
GPU Name:   A10 or A100
CUDA:       >= 11.0
```

Check the **Interconnect** column in results:
- `NVLink` — fast (~200–280 GB/s busbw)
- `PCIe` / `SXM` without NVLink — slower (~25–35 GB/s), fine for dev

**Recommended template**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`
(has CUDA dev headers, gcc 11, Python 3.10 — saves install time)

> Tip: Sort by **$/hr** and filter **On-demand** to avoid spot interruptions.
> 2× A10 PCIe is usually available for ~$0.8–1.2/hr.

Typical prices:

| GPUs | Interconnect | Price |
|---|---|---|
| 2× A10 | PCIe | ~$0.8–1.2/hr |
| 2× A100 | NVLink | ~$2–4/hr |

Click **Rent** → confirm.

---

## 3. Connect via SSH

Vast shows the exact SSH command in **Instances → Connect**. It looks like:

```bash
ssh -p 12345 root@<IP_ADDRESS>
```

The port varies per instance — copy the full command from the dashboard.

---

## 4. What's Pre-installed (PyTorch devel image)

```
CUDA 12.x          ✓
cuDNN              ✓
Python 3.10        ✓
PyTorch            ✓
gcc / g++ 11       ✓
cmake 3.22         ✓
nvidia-smi         ✓
NCCL dev headers   ✓  (included in devel image)
```

**Not pre-installed:**
```
OpenMPI            ✗
matplotlib/numpy   ✗
```

---

## 5. Environment Setup

Run this block once after SSHing in. Takes ~1 minute (less than Lambda — NCCL already present).

```bash
sudo apt update -y

# ── OpenMPI ───────────────────────────────────────────────────────────
sudo apt install -y libopenmpi-dev openmpi-bin

# Verify:
mpirun --version

# ── Python packages ───────────────────────────────────────────────────
pip install matplotlib numpy

# ── Verify NCCL headers are present ──────────────────────────────────
find /usr -name "nccl.h" 2>/dev/null
# Should print something like /usr/include/nccl.h
```

> If NCCL headers are missing (e.g. you chose a non-devel image), install them:
> ```bash
> sudo apt install -y libnccl2 libnccl-dev
> ```

---

## 6. Clone & Build the Project

```bash
# Clone your repo
git clone https://github.com/xuyangwang0825/nccl-comm-profiler.git
cd nccl-comm-profiler

# Check which GPU you have (determines CUDA architecture flag)
nvidia-smi --query-gpu=name --format=csv,noheader
```

**Set the right CUDA architecture** based on your GPU:

| GPU | Architecture flag |
|---|---|
| A100 | `80` |
| A10, RTX 3090 | `86` |
| H100 | `90` |
| V100 | `70` |

```bash
# Build — replace 86 with your GPU's architecture number
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
make -j$(nproc)

# Verify both binaries exist:
ls -lh nccl_bench ring_allreduce
```

---

## 7. Quick Sanity Check Before Full Benchmark

```bash
# Check GPU topology first — confirm P2P / NVLink status
nvidia-smi topo -m
```

Output meaning:
```
      GPU0  GPU1
GPU0   X    NV4      ← NV4 = 4 NVLink connections (fast!)
GPU1  NV4    X

      GPU0  GPU1
GPU0   X    PHB      ← PHB = PCIe via CPU bridge (slower)
GPU1  PHB    X
```

```bash
# Quick ring_allreduce test on 2 GPUs (16 MB)
cd ~/nccl-comm-profiler/build
./ring_allreduce 2 4194304
# Expected: [PASS] with some busbw number
```

---

## 8. Run the Full Benchmark

```bash
cd ~/nccl-comm-profiler

# Run everything (nccl_bench + ring_allreduce + parse + plots)
# Takes about 5-10 minutes
./scripts/run_bench.sh 2

# Results are in:
ls results/
#   nccl_bench.csv        ← NCCL collective benchmarks
#   ring_allreduce.csv    ← custom ring benchmark
#   nccl_debug.log        ← NCCL_DEBUG=INFO topology output
#   topology.json         ← parsed topology summary
#   plots/                ← PNG charts
```

---

## 9. Copy Results Back to Your Mac

```bash
# On your Mac — note the -p flag for the custom port
scp -P 12345 -r root@<IP>:~/nccl-comm-profiler/results ./results_vast

# Or with rsync:
rsync -avz -e "ssh -p 12345" root@<IP>:~/nccl-comm-profiler/results/ ./results_vast/
```

---

## 10. Push Code to GitHub

```bash
# On the instance:
cd ~/nccl-comm-profiler

git config user.email "wangxuyang1998@gmail.com"
git config user.name "Xuyang Wang"

cat > .gitignore << 'EOF'
build/
results/*.log
results/*.csv
EOF

git add .
git commit -m "Add benchmarks and profiler"

# Push with a GitHub personal access token
# Generate one at: github.com → Settings → Developer Settings → Tokens
git remote set-url origin https://<YOUR_TOKEN>@github.com/xuyangwang0825/nccl-comm-profiler.git
git push origin main
```

---

## 11. Destroy the Instance (Important — saves money)

**Instances → Destroy**

> ⚠️ **Destroy** (not just stop) — Vast bills per second even when idle.
> Download results or push to GitHub before destroying.

Cost estimate:
- 2× A10 at ~$1/hr × 2 hours = **~$2** for full benchmarks + exploration
- 2× A100 NVLink at ~$3/hr × 1 hour = **~$3** for NVLink comparison data

---

## Troubleshooting

**`NCCL error: unhandled system type`**
```bash
# NCCL can't find its library:
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**`cmake: NCCL not found`**
```bash
# Check where NCCL is installed:
find /usr -name "nccl.h" 2>/dev/null
find /usr -name "libnccl*" 2>/dev/null
# Pass the paths to cmake:
cmake .. -DNCCL_INCLUDE_DIR=/path/to/include -DNCCL_LIB=/path/to/libnccl.so
```

**`nvcc fatal: Unsupported gpu architecture 'compute_86'`**
```bash
nvcc --version   # check version
# CUDA 11.1+ supports sm_86. Vast's devel image ships CUDA 12, so this shouldn't happen.
# If it does: cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
```

**`std::barrier` compile error**
```bash
g++ --version
# If < 11:
sudo apt install -y g++-11
cmake .. -DCMAKE_CXX_COMPILER=g++-11
```

**Ring allreduce shows `[FAIL]`**
```bash
./ring_allreduce 2 4194304   # watch for [warn] no direct P2P messages
# [FAIL] means wrong values (a bug), not a perf issue.
# Check that count % n_gpus == 0.
```

---

## What to Check in the Results

```bash
# Peak NCCL AllReduce busbw (the headline number for your resume):
grep 'AllReduce' results/nccl_bench.csv | awk -F',' '{print $NF}' | sort -n | tail -3

# NCCL vs ring speedup at 256MB:
grep '268435456' results/nccl_bench.csv
grep '268435456' results/ring_allreduce.csv

# Topology detected:
cat results/topology.json
```

**Expected ranges:**

| Setup | NCCL AllReduce busbw | Ring busbw |
|---|---|---|
| 2× A10 (PCIe) | 25–35 GB/s | 8–15 GB/s |
| 2× A100 (NVLink) | 200–280 GB/s | 60–100 GB/s |

These numbers are what you'll cite in your interview.
