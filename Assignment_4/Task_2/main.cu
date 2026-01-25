#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdint>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
                  << " at " << __FILE__ << ":" << __LINE__         \
                  << std::endl;                                    \
        std::exit(1);                                              \
    }                                                              \
} while(0)

// scan внутри блока (exclusive), 2 элемента на поток
template<int BLOCK>
__global__ void scan_blocks_exclusive(const int* in,
                                      int* out_excl,
                                      int* block_sums,
                                      int n)
{
    __shared__ int temp[2 * BLOCK];

    int tid  = threadIdx.x;
    int base = 2 * BLOCK * blockIdx.x;

    int i0 = base + tid;
    int i1 = base + BLOCK + tid;

    temp[tid]          = (i0 < n) ? in[i0] : 0;
    temp[BLOCK + tid]  = (i1 < n) ? in[i1] : 0;

    // upsweep
    for (int offset = 1; offset < 2 * BLOCK; offset <<= 1) {
        __syncthreads();
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < 2 * BLOCK) temp[idx] += temp[idx - offset];
    }

    // сумма блока + обнуление для exclusive
    __syncthreads();
    if (tid == 0) {
        block_sums[blockIdx.x] = temp[2 * BLOCK - 1];
        temp[2 * BLOCK - 1] = 0;
    }

    // downsweep
    for (int offset = BLOCK; offset > 0; offset >>= 1) {
        __syncthreads();
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < 2 * BLOCK) {
            int t = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += t;
        }
    }
    __syncthreads();

    if (i0 < n) out_excl[i0] = temp[tid];
    if (i1 < n) out_excl[i1] = temp[BLOCK + tid];
}

// scan сумм блоков (exclusive) в 1 блоке
template<int BLOCK>
__global__ void scan_block_sums_exclusive(const int* block_sums,
                                         int* block_offsets,
                                         int numBlocks)
{
    __shared__ int temp[2 * BLOCK];
    int tid = threadIdx.x;

    int i0 = tid;
    int i1 = BLOCK + tid;

    temp[tid]         = (i0 < numBlocks) ? block_sums[i0] : 0;
    temp[BLOCK + tid] = (i1 < numBlocks) ? block_sums[i1] : 0;

    for (int offset = 1; offset < 2 * BLOCK; offset <<= 1) {
        __syncthreads();
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < 2 * BLOCK) temp[idx] += temp[idx - offset];
    }

    __syncthreads();
    if (tid == 0) temp[2 * BLOCK - 1] = 0;

    for (int offset = BLOCK; offset > 0; offset >>= 1) {
        __syncthreads();
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < 2 * BLOCK) {
            int t = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += t;
        }
    }
    __syncthreads();

    if (i0 < numBlocks) block_offsets[i0] = temp[tid];
    if (i1 < numBlocks) block_offsets[i1] = temp[BLOCK + tid];
}

// добавляем оффсет блока и делаем inclusive
template<int BLOCK>
__global__ void add_offsets_make_inclusive(const int* in,
                                           const int* out_excl,
                                           int* out_incl,
                                           const int* block_offsets,
                                           int n)
{
    int tid  = threadIdx.x;
    int base = 2 * BLOCK * blockIdx.x;
    int off  = block_offsets[blockIdx.x];

    int i0 = base + tid;
    int i1 = base + BLOCK + tid;

    if (i0 < n) out_incl[i0] = out_excl[i0] + off + in[i0];
    if (i1 < n) out_incl[i1] = out_excl[i1] + off + in[i1];
}

// CPU inclusive scan
static double cpu_scan(const std::vector<int>& in, std::vector<int>& out) {
    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t s = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        s += in[i];
        out[i] = (int)s;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

int main() {
    const int N = 1'000'000;
    constexpr int BLOCK = 512;                 // 2*BLOCK=1024
    const int elemsPerBlock = 2 * BLOCK;
    const int numBlocks = (N + elemsPerBlock - 1) / elemsPerBlock;

    std::vector<int> h_in(N), h_cpu(N), h_gpu(N);

    std::mt19937 gen(123);
    std::uniform_int_distribution<int> dist(0, 10); // небольшие числа, без переполнения
    for (int i = 0; i < N; ++i) h_in[i] = dist(gen);

    double cpu_ms = cpu_scan(h_in, h_cpu);

    int *d_in=nullptr, *d_excl=nullptr, *d_incl=nullptr, *d_block_sums=nullptr, *d_block_offsets=nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_excl, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_incl, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_sums, numBlocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, numBlocks * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    const int iters = 50;

    // warm-up
    scan_blocks_exclusive<BLOCK><<<numBlocks, BLOCK>>>(d_in, d_excl, d_block_sums, N);
    scan_block_sums_exclusive<BLOCK><<<1, BLOCK>>>(d_block_sums, d_block_offsets, numBlocks);
    add_offsets_make_inclusive<BLOCK><<<numBlocks, BLOCK>>>(d_in, d_excl, d_incl, d_block_offsets, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        scan_blocks_exclusive<BLOCK><<<numBlocks, BLOCK>>>(d_in, d_excl, d_block_sums, N);
        scan_block_sums_exclusive<BLOCK><<<1, BLOCK>>>(d_block_sums, d_block_offsets, numBlocks);
        add_offsets_make_inclusive<BLOCK><<<numBlocks, BLOCK>>>(d_in, d_excl, d_incl, d_block_offsets, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float gpu_ms = total_ms / iters;

    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_incl, N * sizeof(int), cudaMemcpyDeviceToHost));

    int maxDiff = 0;
    for (int i = 0; i < N; ++i) {
        int diff = std::abs(h_gpu[i] - h_cpu[i]);
        if (diff > maxDiff) maxDiff = diff;
    }

    std::cout << "Assignment 4 - Task 2: Prefix Sum (Scan) using Shared Memory\n";
    std::cout << "N = " << N << "\n";
    std::cout << "CPU time (ms) = " << std::fixed << std::setprecision(6) << cpu_ms << "\n";
    std::cout << "GPU time (ms) = " << gpu_ms << " (avg over " << iters << " runs, kernels only)\n";
    std::cout << "Speedup = " << (cpu_ms / gpu_ms) << "x\n";
    std::cout << "Max diff = " << maxDiff << "\n";
    std::cout << "Correct = " << (maxDiff == 0 ? "YES" : "NO") << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_excl));
    CUDA_CHECK(cudaFree(d_incl));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_block_offsets));
    return 0;
}
