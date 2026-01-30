#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <chrono>

#define CUDA_OK(x) do {                                  \
    cudaError_t err = (x);                               \
    if (err != cudaSuccess) {                            \
        std::cerr << "CUDA error: "                      \
                  << cudaGetErrorString(err)             \
                  << " (" << __FILE__ << ":"             \
                  << __LINE__ << ")\n";                  \
        std::exit(1);                                   \
    }                                                    \
} while (0)

// CPU сумма (double просто чтобы эталон был стабильнее)
static double cpu_sum(const float* a, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)a[i];
    return s;
}

// CPU inclusive scan
static void cpu_scan_inclusive(const float* in, float* out, int n)
{
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += (double)in[i];
        out[i] = (float)acc;
    }
}

// Наивная GPU редукция: atomicAdd в глобальную память (обычно медленно)
__global__ void reduce_atomic_kernel(const float* __restrict__ in, float* out, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) atomicAdd(out, in[gid]);
}

// Нормальная GPU редукция через shared memory (каждый блок пишет частичную сумму)
__global__ void reduce_shared_kernel(const float* __restrict__ in,
                                     float* __restrict__ block_sums,
                                     int n)
{
    extern __shared__ float sh[];

    unsigned tid  = threadIdx.x;
    unsigned base = blockIdx.x * (blockDim.x * 2u) + tid;

    float local = 0.0f;
    if (base < (unsigned)n) local += in[base];
    if (base + blockDim.x < (unsigned)n) local += in[base + blockDim.x];

    sh[tid] = local;
    __syncthreads();

    for (unsigned step = blockDim.x / 2u; step > 0; step >>= 1u) {
        if (tid < step) sh[tid] += sh[tid + step];
        __syncthreads();
    }

    if (tid == 0) block_sums[blockIdx.x] = sh[0];
}

// Многошаговая редукция: сжимаем массив, пока не останется один элемент
static float gpu_reduce_shared(const float* d_in, int n, int threads, float& ms_out)
{
    int blocks0 = (n + (threads * 2 - 1)) / (threads * 2);

    float* d_a = nullptr;
    float* d_b = nullptr;
    CUDA_OK(cudaMalloc(&d_a, blocks0 * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_b, blocks0 * sizeof(float)));

    const float* cur_in = d_in;
    float* cur_out = d_a;
    int cur_n = n;

    cudaEvent_t st, en;
    CUDA_OK(cudaEventCreate(&st));
    CUDA_OK(cudaEventCreate(&en));
    CUDA_OK(cudaEventRecord(st));

    bool toggle = false;
    while (true) {
        int cur_blocks = (cur_n + (threads * 2 - 1)) / (threads * 2);

        reduce_shared_kernel<<<cur_blocks, threads, threads * sizeof(float)>>>(cur_in, cur_out, cur_n);
        CUDA_OK(cudaGetLastError());

        if (cur_blocks == 1) break;

        cur_n = cur_blocks;
        cur_in = cur_out;

        toggle = !toggle;
        cur_out = toggle ? d_b : d_a;
    }

    CUDA_OK(cudaEventRecord(en));
    CUDA_OK(cudaEventSynchronize(en));
    CUDA_OK(cudaEventElapsedTime(&ms_out, st, en));

    float res = 0.0f;
    CUDA_OK(cudaMemcpy(&res, cur_out, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_OK(cudaEventDestroy(st));
    CUDA_OK(cudaEventDestroy(en));
    CUDA_OK(cudaFree(d_a));
    CUDA_OK(cudaFree(d_b));

    return res;
}

// Hillis–Steele scan в пределах блока (inclusive), shared memory
__global__ void scan_hillis_kernel(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   float* __restrict__ block_sums,
                                   int n)
{
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    sh[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float val = (tid >= offset) ? sh[tid - offset] : 0.0f;
        __syncthreads();
        sh[tid] += val;
        __syncthreads();
    }

    if (gid < n) out[gid] = sh[tid];

    if (tid == 0) {
        int start = blockIdx.x * blockDim.x;
        int valid = n - start;
        if (valid > blockDim.x) valid = blockDim.x;
        block_sums[blockIdx.x] = (valid > 0) ? sh[valid - 1] : 0.0f;
    }
}

// Blelloch exclusive scan (shared), потом перевод в inclusive
__device__ void blelloch_exclusive(float* s)
{
    int tid = threadIdx.x;
    int n = blockDim.x;

    for (int offset = 1; offset < n; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        __syncthreads();
        if (idx < n) s[idx] += s[idx - offset];
    }

    __syncthreads();
    if (tid == 0) s[n - 1] = 0.0f;

    for (int offset = n >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        __syncthreads();
        if (idx < n) {
            float t = s[idx - offset];
            s[idx - offset] = s[idx];
            s[idx] += t;
        }
    }
    __syncthreads();
}

__global__ void scan_blelloch_kernel(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     float* __restrict__ block_sums,
                                     int n)
{
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float x = (gid < n) ? in[gid] : 0.0f;
    sh[tid] = x;
    __syncthreads();

    blelloch_exclusive(sh);

    if (gid < n) out[gid] = sh[tid] + x;

    if (tid == 0) {
        int start = blockIdx.x * blockDim.x;
        int valid = n - start;
        if (valid > blockDim.x) valid = blockDim.x;

        float total = 0.0f;
        if (valid > 0) {
            int last = valid - 1;
            total = sh[last] + in[start + last];
        }
        block_sums[blockIdx.x] = total;
    }
}

__global__ void add_offsets_kernel(float* out, const float* offsets, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    int b = blockIdx.x;
    if (b > 0) out[gid] += offsets[b - 1];
}

using ScanKernel = void(*)(const float*, float*, float*, int);

// Рекурсивный скан сумм блоков, потом добавляем оффсет к каждому блоку
static void gpu_scan_recursive(float* d_in, float* d_out, int n, int threads, ScanKernel k)
{
    int blocks = (n + threads - 1) / threads;

    float* d_block_sums = nullptr;
    CUDA_OK(cudaMalloc(&d_block_sums, blocks * sizeof(float)));

    k<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, d_block_sums, n);
    CUDA_OK(cudaGetLastError());

    if (blocks > 1) {
        float* d_scanned = nullptr;
        CUDA_OK(cudaMalloc(&d_scanned, blocks * sizeof(float)));

        gpu_scan_recursive(d_block_sums, d_scanned, blocks, threads, k);
        add_offsets_kernel<<<blocks, threads>>>(d_out, d_scanned, n);
        CUDA_OK(cudaGetLastError());

        CUDA_OK(cudaFree(d_scanned));
    }

    CUDA_OK(cudaFree(d_block_sums));
}

static float time_scan(float* d_in, float* d_out, int n, int threads, ScanKernel k)
{
    cudaEvent_t st, en;
    CUDA_OK(cudaEventCreate(&st));
    CUDA_OK(cudaEventCreate(&en));

    CUDA_OK(cudaEventRecord(st));
    gpu_scan_recursive(d_in, d_out, n, threads, k);
    CUDA_OK(cudaEventRecord(en));
    CUDA_OK(cudaEventSynchronize(en));

    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, st, en));

    CUDA_OK(cudaEventDestroy(st));
    CUDA_OK(cudaEventDestroy(en));
    return ms;
}

int main()
{
    const std::vector<int> sizes = { 1024, 1'000'000, 10'000'000 };
    const int threads = 256;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::cout << std::fixed << std::setprecision(6);

    for (int N : sizes) {
        float *h_in = nullptr, *h_ref = nullptr, *h_out = nullptr;

        CUDA_OK(cudaMallocHost(&h_in,  N * sizeof(float)));
        CUDA_OK(cudaMallocHost(&h_ref, N * sizeof(float)));
        CUDA_OK(cudaMallocHost(&h_out, N * sizeof(float)));

        for (int i = 0; i < N; ++i) h_in[i] = dist(rng);

        auto t0 = std::chrono::high_resolution_clock::now();
        double cpuS = cpu_sum(h_in, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_red_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        cpu_scan_inclusive(h_in, h_ref, N);
        t1 = std::chrono::high_resolution_clock::now();
        double cpu_scan_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        float *d_in = nullptr, *d_tmp = nullptr;
        CUDA_OK(cudaMalloc(&d_in,  N * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_tmp, N * sizeof(float)));
        CUDA_OK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

        float red_atomic_ms = 0.0f;
        float red_shared_ms = 0.0f;

        float* d_sum = nullptr;
        CUDA_OK(cudaMalloc(&d_sum, sizeof(float)));
        CUDA_OK(cudaMemset(d_sum, 0, sizeof(float)));

        int blocks = (N + threads - 1) / threads;

        cudaEvent_t st, en;
        CUDA_OK(cudaEventCreate(&st));
        CUDA_OK(cudaEventCreate(&en));
        CUDA_OK(cudaEventRecord(st));
        reduce_atomic_kernel<<<blocks, threads>>>(d_in, d_sum, N);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaEventRecord(en));
        CUDA_OK(cudaEventSynchronize(en));
        CUDA_OK(cudaEventElapsedTime(&red_atomic_ms, st, en));

        float gpu_atomic_sum = 0.0f;
        CUDA_OK(cudaMemcpy(&gpu_atomic_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

        float gpu_shared_sum = gpu_reduce_shared(d_in, N, threads, red_shared_ms);

        float scan_hillis_ms = time_scan(d_in, d_tmp, N, threads, scan_hillis_kernel);
        float scan_blelloch_ms = time_scan(d_in, d_tmp, N, threads, scan_blelloch_kernel);

        CUDA_OK(cudaMemcpy(h_out, d_tmp, N * sizeof(float), cudaMemcpyDeviceToHost));

        double max_err = 0.0;
        for (int i = 0; i < N; ++i) {
            double e = std::fabs((double)h_out[i] - (double)h_ref[i]);
            if (e > max_err) max_err = e;
        }

        double abs_err_atomic = std::fabs((double)gpu_atomic_sum - cpuS);
        double abs_err_shared = std::fabs((double)gpu_shared_sum - cpuS);

        std::cout << "N = " << N << "\n";
        std::cout << "CPU reduction time (ms) = " << cpu_red_ms << "\n";
        std::cout << "CPU scan time (ms) = " << cpu_scan_ms << "\n";
        std::cout << "GPU reduction atomic (ms) = " << red_atomic_ms
                  << ", abs err = " << abs_err_atomic << "\n";
        std::cout << "GPU reduction shared (ms) = " << red_shared_ms
                  << ", abs err = " << abs_err_shared << "\n";
        std::cout << "GPU scan Hillis-Steele (ms) = " << scan_hillis_ms << "\n";
        std::cout << "GPU scan Blelloch (ms) = " << scan_blelloch_ms
                  << ", max abs err = " << max_err << "\n";

        CUDA_OK(cudaFree(d_sum));
        CUDA_OK(cudaFree(d_in));
        CUDA_OK(cudaFree(d_tmp));

        CUDA_OK(cudaFreeHost(h_in));
        CUDA_OK(cudaFreeHost(h_ref));
        CUDA_OK(cudaFreeHost(h_out));

        CUDA_OK(cudaEventDestroy(st));
        CUDA_OK(cudaEventDestroy(en));
    }

    return 0;
}
