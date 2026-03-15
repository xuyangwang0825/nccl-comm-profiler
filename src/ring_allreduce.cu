/**
 * ring_allreduce.cu
 *
 * Hand-rolled ring-allreduce on a single node using CUDA peer-to-peer
 * transfers and std::barrier for inter-thread synchronization.
 *
 * One CPU thread controls one GPU. Threads communicate via a shared
 * RingState struct that holds pointers to each GPU's device buffers.
 *
 * Algorithm (N GPUs, data split into N chunks of `chunk` floats):
 *
 *   Phase 1 – Reduce-Scatter (N-1 steps):
 *     Each step: GPU i pushes one chunk into GPU (i+1)'s recv buffer,
 *                then adds the chunk it received into its own data buffer.
 *     After N-1 steps: GPU i holds the fully-reduced sum of chunk i.
 *
 *   Phase 2 – AllGather (N-1 steps):
 *     Each step: GPU i copies its reduced chunk to GPU (i+1)'s data buffer.
 *     After N-1 steps: all GPUs hold all fully-reduced chunks.
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. && make ring_allreduce
 *
 * Run:
 *   ./build/ring_allreduce [n_gpus] [count]
 *   ./build/ring_allreduce 4 4194304    # 4 GPUs, 16 MB of floats
 */

#include <cuda_runtime.h>

#include <barrier>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <vector>
#include <chrono>

// ─── Error checking macros ──────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ─── CUDA kernel: element-wise in-place addition  dst[i] += src[i] ─────────

__global__ void add_kernel(float* __restrict__ dst,
                           const float* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// ─── Shared state visible to all GPU threads ────────────────────────────────

struct RingState {
    int n_gpus;
    int count;                   // total number of floats per GPU
    std::vector<float*> d_data;  // d_data[g] = device buffer on GPU g  (size: count)
    std::vector<float*> d_recv;  // d_recv[g] = temp recv buffer on GPU g (size: count)
    // d_recv exists so writes from neighboring GPUs don't corrupt d_data mid-step.
};

// ─── Per-GPU worker thread ───────────────────────────────────────────────────
//
// Each thread:
//   1. Sets its CUDA device context
//   2. Enables P2P (peer-to-peer) access to all other GPUs
//   3. Runs Reduce-Scatter
//   4. Runs AllGather
//
// std::barrier ensures all threads reach the same point before any
// transfers happen, preventing a fast GPU from reading stale data.

void ring_worker(int rank, RingState* st, std::barrier<>* bar) {
    CUDA_CHECK(cudaSetDevice(rank));

    const int N     = st->n_gpus;
    const int chunk = st->count / N;  // floats per chunk

    // ── Enable P2P access ────────────────────────────────────────────────
    // Without P2P, cudaMemcpyPeerAsync falls back to staging through CPU RAM.
    // With NVLink, P2P gives direct GPU-GPU transfers at ~300 GB/s (A100).
    for (int peer = 0; peer < N; peer++) {
        if (peer == rank) continue;
        int can_access = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, rank, peer));
        if (can_access) {
            cudaDeviceEnablePeerAccess(peer, 0);  // may return "already enabled" error, ignore
        } else {
            // PCIe fallback is automatic; just note it for the user
            if (rank == 0)
                printf("[warn] GPU %d <-> GPU %d: no direct P2P (using PCIe staging)\n",
                       rank, peer);
        }
    }
    bar->arrive_and_wait();  // ── barrier: all peer accesses enabled ────────

    // ── PHASE 1: Reduce-Scatter ──────────────────────────────────────────
    //
    // In step `s`:
    //   send_chunk  = (rank - s      + N) % N   ← chunk we push to next GPU
    //   recv_chunk  = (rank - s - 1  + N) % N   ← chunk we receive from prev GPU
    //   send_to     = (rank + 1) % N            ← always push clockwise
    //
    // We write into the NEXT GPU's d_recv buffer (not d_data) to avoid
    // the race where a GPU reads a chunk before the sender finishes writing.
    // After all transfers complete (barrier), we reduce d_recv → d_data.

    for (int step = 0; step < N - 1; step++) {
        const int send_to    = (rank + 1) % N;
        const int send_chunk = (rank - step + N) % N;
        const int recv_chunk = (rank - step - 1 + N) % N;

        bar->arrive_and_wait();  // all threads ready to transfer

        // Push our chunk into the next GPU's recv slot for that chunk index
        CUDA_CHECK(cudaMemcpyPeerAsync(
            st->d_recv[send_to] + (size_t)recv_chunk * chunk,  // dst ptr on next GPU
            send_to,                                             // dst device
            st->d_data[rank]   + (size_t)send_chunk * chunk,   // src ptr on this GPU
            rank,                                               // src device
            (size_t)chunk * sizeof(float)
        ));
        CUDA_CHECK(cudaDeviceSynchronize());  // wait for OUR copy to land

        bar->arrive_and_wait();  // all copies landed; safe to reduce

        // Element-wise add: d_data[recv_chunk] += d_recv[recv_chunk]
        // This accumulates the partial sum from the previous GPU.
        const int tpb = 256;
        add_kernel<<<(chunk + tpb - 1) / tpb, tpb>>>(
            st->d_data[rank] + (size_t)recv_chunk * chunk,
            st->d_recv[rank] + (size_t)recv_chunk * chunk,
            chunk
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        bar->arrive_and_wait();  // all reductions done; next step can start
    }

    // After reduce-scatter:
    //   GPU rank owns the fully-reduced sum of chunk[rank].

    // ── PHASE 2: AllGather ───────────────────────────────────────────────
    //
    // In step `s`:
    //   send_chunk = (rank - s + N) % N   ← chunk we broadcast clockwise
    //   send_to    = (rank + 1) % N
    //
    // This time we write directly into d_data (no addition, just copy),
    // because these chunks are already final. Different steps touch different
    // chunk indices so there's no write conflict.

    for (int step = 0; step < N - 1; step++) {
        const int send_to    = (rank + 1) % N;
        const int send_chunk = (rank - step + N) % N;

        bar->arrive_and_wait();  // all ready to transfer

        CUDA_CHECK(cudaMemcpyPeerAsync(
            st->d_data[send_to] + (size_t)send_chunk * chunk,  // dst (final destination)
            send_to,
            st->d_data[rank]   + (size_t)send_chunk * chunk,   // src
            rank,
            (size_t)chunk * sizeof(float)
        ));
        CUDA_CHECK(cudaDeviceSynchronize());

        bar->arrive_and_wait();  // all copies done; next step can start
    }
    // Every GPU now holds all N fully-reduced chunks.
}

// ─── Verification helper ────────────────────────────────────────────────────
// Each GPU g initializes data[i] = (g+1) * 1.0f for all i.
// After allreduce (sum), expected value = sum(1..N) = N*(N+1)/2.

bool verify(const std::vector<float*>& d_data, int count, int n_gpus) {
    float expected = (float)(n_gpus * (n_gpus + 1)) / 2.0f;
    std::vector<float> h_data(count);
    bool ok = true;
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data[g],
                              count * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < std::min(count, 16); i++) {
            if (fabsf(h_data[i] - expected) > 1e-3f) {
                printf("[FAIL] GPU %d data[%d] = %.2f, expected %.2f\n",
                       g, i, h_data[i], expected);
                ok = false;
                break;
            }
        }
    }
    return ok;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // Parse arguments
    int n_gpus = (argc > 1) ? atoi(argv[1]) : 2;
    int count  = (argc > 2) ? atoi(argv[2]) : 4 * 1024 * 1024;  // 16 MB default

    // Validate
    int avail = 0;
    CUDA_CHECK(cudaGetDeviceCount(&avail));
    if (n_gpus > avail) {
        fprintf(stderr, "Requested %d GPUs but only %d available.\n", n_gpus, avail);
        return 1;
    }
    if (count % n_gpus != 0) {
        fprintf(stderr, "count (%d) must be divisible by n_gpus (%d).\n", count, n_gpus);
        return 1;
    }

    printf("Ring AllReduce: %d GPUs, %d floats (%.1f MB)\n",
           n_gpus, count, count * sizeof(float) / 1e6);

    // ── Allocate buffers on each GPU ────────────────────────────────────
    RingState state;
    state.n_gpus = n_gpus;
    state.count  = count;
    state.d_data.resize(n_gpus);
    state.d_recv.resize(n_gpus);

    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMalloc(&state.d_data[g], (size_t)count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_recv[g], (size_t)count * sizeof(float)));

        // Initialize: GPU g fills data with (g+1) so we can verify the sum
        std::vector<float> h_init(count, (float)(g + 1));
        CUDA_CHECK(cudaMemcpy(state.d_data[g], h_init.data(),
                              count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(state.d_recv[g], 0, (size_t)count * sizeof(float)));
    }

    // ── Run ring-allreduce with one thread per GPU ───────────────────────
    const int WARMUP = 3;
    const int ITERS  = 10;

    std::barrier bar(n_gpus);

    // Warmup
    for (int w = 0; w < WARMUP; w++) {
        // Reinitialize data for each run
        for (int g = 0; g < n_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            std::vector<float> h_init(count, (float)(g + 1));
            CUDA_CHECK(cudaMemcpy(state.d_data[g], h_init.data(),
                                  count * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(state.d_recv[g], 0, (size_t)count * sizeof(float)));
        }
        std::vector<std::thread> threads;
        for (int g = 0; g < n_gpus; g++)
            threads.emplace_back(ring_worker, g, &state, &bar);
        for (auto& t : threads) t.join();
    }

    // Benchmark
    double total_us = 0.0;
    for (int iter = 0; iter < ITERS; iter++) {
        for (int g = 0; g < n_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            std::vector<float> h_init(count, (float)(g + 1));
            CUDA_CHECK(cudaMemcpy(state.d_data[g], h_init.data(),
                                  count * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(state.d_recv[g], 0, (size_t)count * sizeof(float)));
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        for (int g = 0; g < n_gpus; g++)
            threads.emplace_back(ring_worker, g, &state, &bar);
        for (auto& t : threads) t.join();

        auto t1 = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    // ── Verify correctness ───────────────────────────────────────────────
    bool correct = verify(state.d_data, count, n_gpus);

    // ── Report performance ───────────────────────────────────────────────
    double avg_us  = total_us / ITERS;
    double size_gb = (double)count * sizeof(float) / 1e9;

    // Algorithm bandwidth: data_size / time
    double algbw = size_gb / (avg_us / 1e6);
    // Bus bandwidth: accounts for the 2*(N-1)/N factor of the ring pattern
    // This is what to compare against the physical link speed (NVLink/PCIe)
    double busbw = algbw * 2.0 * (n_gpus - 1) / n_gpus;

    printf("\n%-20s %10s %12s %12s %12s\n",
           "impl", "size(MB)", "time(us)", "algbw(GB/s)", "busbw(GB/s)");
    printf("%-20s %10.1f %12.1f %12.2f %12.2f   %s\n",
           "ring_allreduce",
           size_gb * 1000,
           avg_us,
           algbw,
           busbw,
           correct ? "[PASS]" : "[FAIL]");

    // ── Sweep across message sizes and print CSV ─────────────────────────
    printf("\n# CSV for plotting (ring_allreduce)\n");
    printf("size_bytes,time_us,algbw_GBps,busbw_GBps\n");

    int sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576,
                   4194304, 16777216, 67108864, 268435456};

    for (int sz : sizes) {
        if (sz % n_gpus != 0) continue;

        // Reallocate for this size
        for (int g = 0; g < n_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaFree(state.d_data[g]));
            CUDA_CHECK(cudaFree(state.d_recv[g]));
            CUDA_CHECK(cudaMalloc(&state.d_data[g], (size_t)sz * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&state.d_recv[g], (size_t)sz * sizeof(float)));
        }
        state.count = sz;

        double tot = 0.0;
        for (int iter = 0; iter < ITERS; iter++) {
            for (int g = 0; g < n_gpus; g++) {
                CUDA_CHECK(cudaSetDevice(g));
                std::vector<float> h(sz, (float)(g + 1));
                CUDA_CHECK(cudaMemcpy(state.d_data[g], h.data(),
                                      sz * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(state.d_recv[g], 0, (size_t)sz * sizeof(float)));
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<std::thread> threads;
            for (int g = 0; g < n_gpus; g++)
                threads.emplace_back(ring_worker, g, &state, &bar);
            for (auto& t : threads) t.join();
            auto t1 = std::chrono::high_resolution_clock::now();
            tot += std::chrono::duration<double, std::micro>(t1 - t0).count();
        }

        double us    = tot / ITERS;
        double gb    = (double)sz * sizeof(float) / 1e9;
        double alg   = gb / (us / 1e6);
        double bus   = alg * 2.0 * (n_gpus - 1) / n_gpus;
        printf("%d,%.1f,%.3f,%.3f\n", sz * (int)sizeof(float), us, alg, bus);
    }

    // ── Cleanup ──────────────────────────────────────────────────────────
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaFree(state.d_data[g]));
        CUDA_CHECK(cudaFree(state.d_recv[g]));
    }
    return 0;
}
