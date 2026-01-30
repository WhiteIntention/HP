#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

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

// CPU-версия префиксной суммы (inclusive)
// Нужна только для проверки корректности GPU
static void cpu_prefix_sum(const std::vector<float>& in,
                           std::vector<float>& out)
{
    out.resize(in.size());
    double acc = 0.0;
    for (size_t i = 0; i < in.size(); ++i) {
        acc += (double)in[i];
        out[i] = (float)acc;
    }
}

// Реализация Blelloch exclusive scan внутри одного блока
// Работает в shared memory, blockDim.x — степень двойки
__device__ void blelloch_exclusive_scan(float* s)
{
    int tid = threadIdx.x;
    int n   = blockDim.x;

    // Фаза накопления (up-sweep)
    for (int offset = 1; offset < n; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        __syncthreads();
        if (idx < n) {
            s[idx] += s[idx - offset];
        }
    }

    // Последний элемент обнуляем
    __syncthreads();
    if (tid == 0) {
        s[n - 1] = 0.0f;
    }

    // Фаза распространения (down-sweep)
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

// Ядро: считает prefix sum внутри блока
// Параллельно сохраняет сумму элементов блока
__global__ void block_prefix_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    float* __restrict__ block_sums,
                                    int n)
{
    extern __shared__ float sh[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Загружаем элемент или 0, если вышли за границы
    float x = (gid < n) ? in[gid] : 0.0f;
    sh[tid] = x;
    __syncthreads();

    // Exclusive scan в shared memory
    blelloch_exclusive_scan(sh);

    // Переводим exclusive в inclusive
    float inclusive = sh[tid] + x;
    if (gid < n) {
        out[gid] = inclusive;
    }

    // Считаем сумму элементов блока
    __syncthreads();
    if (tid == 0) {
        int block_start = blockIdx.x * blockDim.x;
        int valid = n - block_start;
        if (valid > blockDim.x) valid = blockDim.x;

        float total = 0.0f;
        if (valid > 0) {
            int last = valid - 1;
            total = sh[last] + in[block_start + last];
        }
        block_sums[blockIdx.x] = total;
    }
}

// Ядро добавляет оффсет (сумму предыдущих блоков)
// к каждому элементу текущего блока
__global__ void add_block_offsets(float* __restrict__ out,
                                  const float* __restrict__ block_offsets,
                                  int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    int b = blockIdx.x;
    float offset = (b == 0) ? 0.0f : block_offsets[b - 1];
    out[gid] += offset;
}

// Рекурсивная реализация inclusive scan на GPU
static void gpu_prefix_sum(float* d_in,
                           float* d_out,
                           int n,
                           int threads)
{
    int blocks = (n + threads - 1) / threads;

    float* d_block_sums = nullptr;
    CUDA_OK(cudaMalloc(&d_block_sums, blocks * sizeof(float)));

    block_prefix_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        d_in, d_out, d_block_sums, n
    );
    CUDA_OK(cudaGetLastError());

    if (blocks > 1) {
        float* d_scanned_blocks = nullptr;
        CUDA_OK(cudaMalloc(&d_scanned_blocks, blocks * sizeof(float)));

        // Сканируем суммы блоков
        gpu_prefix_sum(d_block_sums, d_scanned_blocks, blocks, threads);

        // Добавляем оффсеты ко всем элементам
        add_block_offsets<<<blocks, threads>>>(d_out, d_scanned_blocks, n);
        CUDA_OK(cudaGetLastError());

        CUDA_OK(cudaFree(d_scanned_blocks));
    }

    CUDA_OK(cudaFree(d_block_sums));
}

int main()
{
    const std::vector<int> tests = { 1024, 1'000'000, 10'000'000 };
    const int threads = 256;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::cout << std::fixed << std::setprecision(6);

    for (int N : tests) {
        std::vector<float> h_in(N);
        for (int i = 0; i < N; ++i) {
            h_in[i] = dist(rng);
        }

        std::vector<float> h_ref;
        cpu_prefix_sum(h_in, h_ref);

        float *d_in = nullptr, *d_out = nullptr;
        CUDA_OK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_out, N * sizeof(float)));
        CUDA_OK(cudaMemcpy(d_in, h_in.data(),
                           N * sizeof(float),
                           cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CUDA_OK(cudaEventCreate(&start));
        CUDA_OK(cudaEventCreate(&stop));
        CUDA_OK(cudaEventRecord(start));

        gpu_prefix_sum(d_in, d_out, N, threads);

        CUDA_OK(cudaEventRecord(stop));
        CUDA_OK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&ms, start, stop));

        std::vector<float> h_out(N);
        CUDA_OK(cudaMemcpy(h_out.data(), d_out,
                           N * sizeof(float),
                           cudaMemcpyDeviceToHost));

        double max_err = 0.0;
        for (int i = 0; i < N; ++i) {
            double e = std::fabs((double)h_out[i] - (double)h_ref[i]);
            if (e > max_err) max_err = e;
        }

        std::cout << "N = " << N << "\n";
        std::cout << "Max abs error = " << max_err << "\n";
        std::cout << "GPU scan time (ms) = " << ms << "\n";

        CUDA_OK(cudaFree(d_in));
        CUDA_OK(cudaFree(d_out));
        CUDA_OK(cudaEventDestroy(start));
        CUDA_OK(cudaEventDestroy(stop));
    }

    return 0;
}
