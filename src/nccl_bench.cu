/**
 * nccl_bench.cu
 *
 * Benchmarks all four NCCL collective operations across a sweep of
 * message sizes. Outputs CSV that matches the nccl-tests format so
 * the same Python plotter works for both.
 *
 * Operations benchmarked:
 *   AllReduce     – every GPU gets the sum of all GPUs' data
 *   AllGather     – every GPU gets the concatenation of all GPUs' data
 *   ReduceScatter – each GPU gets one fully-reduced shard
 *   Broadcast     – GPU 0 sends its data to all GPUs
 *
 * Timing uses CUDA events placed on each GPU's stream; we take the MAX
 * elapsed time across all GPUs (the slowest GPU defines latency, just
 * like in real distributed training).
 *
 * Bus bandwidth formula (same as nccl-tests):
 *   AllReduce    : busbw = algbw * 2 * (N-1) / N
 *   AllGather    : busbw = algbw * (N-1) / N
 *   ReduceScatter: busbw = algbw * (N-1) / N
 *   Broadcast    : busbw = algbw * (N-1) / N
 *
 * Build: cmake .. && make nccl_bench
 * Run:   ./build/nccl_bench [n_gpus]
 */

#include <nccl.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ─── Error checking macros ──────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t e = (call);                                             \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(e));             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#define NCCL_CHECK(call)                                                    \
    do {                                                                    \
        ncclResult_t r = (call);                                            \
        if (r != ncclSuccess) {                                             \
            fprintf(stderr, "NCCL error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, ncclGetErrorString(r));             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ─── Collective type tag ─────────────────────────────────────────────────────

enum class CollOp { AllReduce, AllGather, ReduceScatter, Broadcast };

const char* op_name(CollOp op) {
    switch (op) {
        case CollOp::AllReduce:     return "AllReduce";
        case CollOp::AllGather:     return "AllGather";
        case CollOp::ReduceScatter: return "ReduceScatter";
        case CollOp::Broadcast:     return "Broadcast";
    }
    return "?";
}

// busbw multiplier differs by operation (from nccl-tests source)
double busbw_factor(CollOp op, int n) {
    switch (op) {
        case CollOp::AllReduce:     return 2.0 * (n - 1) / n;
        case CollOp::AllGather:
        case CollOp::ReduceScatter:
        case CollOp::Broadcast:     return (double)(n - 1) / n;
    }
    return 1.0;
}

// ─── Benchmark one (op, count) pair ─────────────────────────────────────────
//
// Returns average time in microseconds across ITERS iterations.
// Uses per-GPU CUDA events; returns max elapsed across all GPUs.

double bench_one(CollOp op, size_t count, int n_gpus,
                 const std::vector<ncclComm_t>& comms,
                 const std::vector<float*>& d_src,
                 const std::vector<float*>& d_dst,
                 const std::vector<cudaStream_t>& streams,
                 int warmup, int iters) {

    std::vector<cudaEvent_t> ev_start(n_gpus), ev_stop(n_gpus);
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaEventCreate(&ev_start[g]));
        CUDA_CHECK(cudaEventCreate(&ev_stop[g]));
    }

    auto run_collective = [&]() {
        // ncclGroupStart/End submits all GPU operations atomically.
        // Without this, ops are submitted sequentially and the first GPU
        // may finish before the last has even started.
        NCCL_CHECK(ncclGroupStart());
        for (int g = 0; g < n_gpus; g++) {
            switch (op) {
                case CollOp::AllReduce:
                    // sendbuf, recvbuf, count, type, reduce_op, comm, stream
                    NCCL_CHECK(ncclAllReduce(d_src[g], d_dst[g], count,
                                            ncclFloat, ncclSum, comms[g], streams[g]));
                    break;

                case CollOp::AllGather:
                    // Each GPU sends `count` floats; output is count*N floats.
                    // We use d_src as input (count floats) and d_dst as output
                    // (must be allocated to count*N floats; see main).
                    NCCL_CHECK(ncclAllGather(d_src[g], d_dst[g], count,
                                            ncclFloat, comms[g], streams[g]));
                    break;

                case CollOp::ReduceScatter:
                    // Each GPU contributes count*N floats, receives count floats.
                    // We use d_dst as input (count*N) and d_src as output (count).
                    NCCL_CHECK(ncclReduceScatter(d_dst[g], d_src[g], count,
                                                ncclFloat, ncclSum, comms[g], streams[g]));
                    break;

                case CollOp::Broadcast:
                    // GPU 0 (root=0) sends to all GPUs.
                    NCCL_CHECK(ncclBroadcast(d_src[g], d_dst[g], count,
                                            ncclFloat, 0, comms[g], streams[g]));
                    break;
            }
        }
        NCCL_CHECK(ncclGroupEnd());
    };

    // Warmup – lets NCCL do its internal setup (algorithm selection, buffer
    // pinning, connection establishment). Without warmup, first iteration
    // includes one-time setup costs that inflate the latency measurement.
    for (int w = 0; w < warmup; w++) run_collective();
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamSynchronize(streams[g]));
    }

    // Record start events on all streams BEFORE submitting work
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaEventRecord(ev_start[g], streams[g]));
    }

    // Timed iterations
    for (int i = 0; i < iters; i++) run_collective();

    // Record stop events
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaEventRecord(ev_stop[g], streams[g]));
    }

    // Wait for all GPUs and find the MAX elapsed (slowest GPU = actual latency)
    float max_ms = 0.0f;
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaEventSynchronize(ev_stop[g]));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start[g], ev_stop[g]));
        max_ms = std::max(max_ms, ms);
    }

    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaEventDestroy(ev_start[g]));
        CUDA_CHECK(cudaEventDestroy(ev_stop[g]));
    }

    return (double)max_ms * 1000.0 / iters;  // convert ms→us, per iteration
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int n_gpus = (argc > 1) ? atoi(argv[1]) : 2;

    int avail = 0;
    CUDA_CHECK(cudaGetDeviceCount(&avail));
    if (n_gpus > avail) {
        fprintf(stderr, "Requested %d GPUs but only %d available.\n", n_gpus, avail);
        return 1;
    }

    printf("NCCL Collective Benchmark: %d GPUs\n\n", n_gpus);

    // ── Initialize NCCL communicators ────────────────────────────────────
    // ncclCommInitAll is the single-node convenience API.
    // For multi-node you'd use ncclGetUniqueId + ncclCommInitRank.
    std::vector<int> devs(n_gpus);
    for (int g = 0; g < n_gpus; g++) devs[g] = g;

    std::vector<ncclComm_t> comms(n_gpus);
    NCCL_CHECK(ncclCommInitAll(comms.data(), n_gpus, devs.data()));

    // ── Create streams (one per GPU) ─────────────────────────────────────
    // Streams allow ops on different GPUs to overlap in time.
    std::vector<cudaStream_t> streams(n_gpus);
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamCreate(&streams[g]));
    }

    // ── Allocate send/recv buffers ────────────────────────────────────────
    // MAX_COUNT covers the largest message size we'll test.
    // AllGather output is N times larger than input, so d_dst is MAX_COUNT*N.
    const size_t MAX_COUNT = 256 * 1024 * 1024 / sizeof(float);  // 256 MB
    std::vector<float*> d_src(n_gpus), d_dst(n_gpus);
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMalloc(&d_src[g], MAX_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dst[g], MAX_COUNT * n_gpus * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_src[g], 1, MAX_COUNT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dst[g], 0, MAX_COUNT * n_gpus * sizeof(float)));
    }

    // ── Message sizes to sweep ────────────────────────────────────────────
    // Matches nccl-tests default sweep: 8B → 256MB in powers of 2
    std::vector<size_t> sizes;
    for (size_t s = 8; s <= 256 * 1024 * 1024; s *= 2)
        sizes.push_back(s);

    const int WARMUP = 5;
    const int ITERS  = 20;

    // ── Print CSV header ──────────────────────────────────────────────────
    printf("# op,size_bytes,count,time_us,algbw_GBps,busbw_GBps\n");

    for (CollOp op : {CollOp::AllReduce, CollOp::AllGather,
                      CollOp::ReduceScatter, CollOp::Broadcast}) {
        for (size_t sz : sizes) {
            // `count` = number of floats for the send buffer.
            // For AllGather: each GPU sends `count` floats, receives count*N.
            // For ReduceScatter: each GPU sends count*N, receives `count`.
            size_t count = sz / sizeof(float);
            if (count == 0) continue;
            if (count > MAX_COUNT) break;

            double us = bench_one(op, count, n_gpus, comms, d_src, d_dst,
                                  streams, WARMUP, ITERS);

            // algbw = data_GB / time_s  (using the send buffer size as "data")
            double data_gb = (double)sz / 1e9;
            double algbw   = data_gb / (us / 1e6);
            double busbw   = algbw * busbw_factor(op, n_gpus);

            printf("%s,%zu,%zu,%.2f,%.3f,%.3f\n",
                   op_name(op), sz, count, us, algbw, busbw);
            fflush(stdout);
        }
    }

    // ── Cleanup ──────────────────────────────────────────────────────────
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaFree(d_src[g]));
        CUDA_CHECK(cudaFree(d_dst[g]));
        CUDA_CHECK(cudaStreamDestroy(streams[g]));
        NCCL_CHECK(ncclCommDestroy(comms[g]));
    }

    return 0;
}
