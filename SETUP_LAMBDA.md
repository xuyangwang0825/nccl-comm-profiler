# Lambda Cloud Setup Guide

How to spin up a 2-GPU instance on Lambda Labs and run this profiler end-to-end.

> **Lambda out of capacity?** See [SETUP_VAST.md](SETUP_VAST.md) for Vast.ai instructions.

---

## 1. Create Account & Add SSH Key

1. Go to [lambdalabs.com](https://lambdalabs.com) → Sign up
2. **Dashboard → SSH Keys → Add SSH Key**
   ```bash
   # On your Mac, generate a key if you don't have one
   ssh-keygen -t ed25519 -C "lambda"
   cat ~/.ssh/id_ed25519.pub   # copy this output into Lambda's SSH key field
   ```

---

## 2. Launch a 2-GPU Instance

**Dashboard → Instances → Launch Instance**

Recommended instance types (check availability — Lambda stock fluctuates):

| Instance | GPUs | Interconnect | Price | CUDA arch |
|---|---|---|---|---|
| `gpu_8x_a100` | 8× A100 SXM4 | **NVLink** | ~$10/hr | `80` |
| `gpu_2x_a10`  | 2× A10       | PCIe only  | ~$1.5/hr | `86` |
| `gpu_1x_a100` | 1× A100      | n/a        | ~$1.3/hr | `80` |

> **Recommendation**: Start with `gpu_2x_a10` (cheapest 2-GPU option). It's PCIe
> only (no NVLink), so your busbw will be lower (~30 GB/s), but it's perfect for
> learning and testing the code. If you want the NVLink comparison data for the
> resume, grab `gpu_8x_a100` for 1-2 hours.

Select:
- **Region**: US-West or US-TX (pick whichever has stock)
- **Filesystem**: None needed for this project
- Click **Launch** → confirm

---

## 3. Connect via SSH

Lambda shows you the IP after the instance starts (takes ~1 min).

```bash
# Replace with your actual instance IP
ssh ubuntu@<INSTANCE_IP>

# Or add to ~/.ssh/config for convenience:
# Host lambda
#     HostName <INSTANCE_IP>
#     User ubuntu
#     IdentityFile ~/.ssh/id_ed25519
# Then just: ssh lambda
```

---

## 4. What's Pre-installed (Lambda already set up for you)

Lambda's base image includes:
```
CUDA 12.x          ✓  (check with: nvcc --version)
cuDNN              ✓
Python 3.10        ✓
PyTorch            ✓
nvidia-smi         ✓
gcc / g++ 11       ✓  (supports C++20 for std::barrier)
```

**Not pre-installed** (you need to install):
```
NCCL dev headers   ✗
CMake >= 3.18      ✗  (system cmake is usually 3.22, check first)
OpenMPI            ✗
matplotlib/numpy   ✗
```

---

## 5. Environment Setup

Run this block once after SSHing in. Takes ~3 minutes.

```bash
# ── Update package list ──────────────────────────────────────────────
sudo apt update -y

# ── NCCL (library + dev headers) ─────────────────────────────────────
# Lambda's CUDA is in /usr/local/cuda, so we use the CUDA repo NCCL
sudo apt install -y libnccl2 libnccl-dev

# Verify NCCL is installed:
dpkg -l | grep nccl
# Should show libnccl2 and libnccl-dev

# ── OpenMPI ───────────────────────────────────────────────────────────
# MPI is used for multi-node. For single-node we only need the headers.
sudo apt install -y libopenmpi-dev openmpi-bin

# Verify:
mpirun --version

# ── CMake ─────────────────────────────────────────────────────────────
cmake --version   # if >= 3.18, you're good. Otherwise:
# sudo apt install -y cmake   # or pip install cmake

# ── Python packages ───────────────────────────────────────────────────
pip install matplotlib numpy

# ── Git (already installed, just in case) ─────────────────────────────
git --version
```

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
make -j$(nproc)     # nproc = number of CPU cores, speeds up compilation

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
# On your Mac (not on the instance):
scp -r ubuntu@<INSTANCE_IP>:~/nccl-comm-profiler/results ./results_lambda

# Or with rsync (faster, skips unchanged files):
rsync -avz ubuntu@<INSTANCE_IP>:~/nccl-comm-profiler/results/ ./results_lambda/
```

---

## 10. Push Code to GitHub

```bash
# On the instance:
cd ~/nccl-comm-profiler

git config user.email "wangxuyang1998@gmail.com"
git config user.name "Xuyang Wang"

# Add a .gitignore first
cat > .gitignore << 'EOF'
build/
results/*.log
results/*.csv
# Keep plots in results/plots/ if you want them in the repo
EOF

git add .
git commit -m "Add benchmarks and profiler"

# Push — you'll need a GitHub personal access token since SSH keys
# on the instance are different from your GitHub SSH key
# Generate one at: github.com → Settings → Developer Settings → Tokens
git remote set-url origin https://<YOUR_TOKEN>@github.com/xuyangwang0825/nccl-comm-profiler.git
git push origin main
```

---

## 11. Terminate the Instance (Important — saves money)

**Dashboard → Instances → Terminate**

> ⚠️ **Terminate** (not just stop) — Lambda charges per hour even when idle.
> Your files are lost on termination, so push to GitHub or download results first.

Cost estimate:
- 2× A10 at $1.5/hr × 2 hours = **~$3** for full benchmarks + exploration
- 8× A100 at $10/hr × 1 hour = **~$10** for NVLink comparison data

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
# Your CUDA version is too old for that architecture.
nvcc --version   # check version
# CUDA 11.1+ supports sm_86. Lambda should have CUDA 12, so this shouldn't happen.
# If it does: cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
```

**`ring_allreduce` binary missing after build / `<barrier>` not found**
```bash
# ring_allreduce.cu uses std::barrier (C++20). Wipe the build dir and rebuild —
# the repo sets CMAKE_CUDA_STANDARD 20 which fixes this:
cd ~/nccl-comm-profiler && rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
make -j$(nproc)
```

**`std::barrier` compile error**
```bash
# Need gcc 11+ for C++20 std::barrier
g++ --version
# If < 11:
sudo apt install -y g++-11
cmake .. -DCMAKE_CXX_COMPILER=g++-11
```

**Ring allreduce shows `[FAIL]`**
```bash
# Likely P2P access issue. Check:
./ring_allreduce 2 4194304   # watch for [warn] no direct P2P messages
# If PCIe only, P2P may still work via staging — [FAIL] means wrong values,
# which indicates a bug. Open an issue or check that count % n_gpus == 0.
```

---

## What to Check in the Results

After running, look for these numbers:

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
| 8× A100 (NVLink) | 200–280 GB/s | 60–100 GB/s |

These numbers are what you'll cite in your interview.
