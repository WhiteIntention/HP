#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <omp.h>

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

// GPU: умножаем элементы на 2
__global__ void mul2_kernel(float* data, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) data[gid] *= 2.0f;
}

// CPU: умножаем элементы на 2 (OpenMP)
static void cpu_mul2_omp(float* data, int n)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        data[i] *= 2.0f;
    }
}

int main()
{
    const std::vector<int> sizes = { 1'000'000, 10'000'000 };

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::cout << std::fixed << std::setprecision(6);

    for (int N : sizes) {
        std::vector<float> h_data(N);
        for (int i = 0; i < N; ++i) h_data[i] = dist(rng);

        int half = N / 2;
        int cpu_n = half;
        int gpu_n = N - half;

        // Выделяем GPU память только под вторую половину
        float* d_part = nullptr;
        CUDA_OK(cudaMalloc(&d_part, gpu_n * sizeof(float)));

        int threads = 256;
        int blocks = (gpu_n + threads - 1) / threads;

        // Нам важно измерить общее время гибридной обработки
        // Поэтому стартуем таймер до запуска CPU+GPU и останавливаем после синхронизации
        auto t0 = std::chrono::high_resolution_clock::now();

        // Чтобы CPU и GPU реально работали одновременно:
        // - CPU работает в отдельном потоке OpenMP
        // - GPU запускаем в главном потоке (асинхронно через stream)
        cudaStream_t stream;
        CUDA_OK(cudaStreamCreate(&stream));

        // Копирование и kernel на GPU запускаем асинхронно
        // (пока GPU занят, CPU параллельно умножает свою часть)
        CUDA_OK(cudaMemcpyAsync(d_part,
                                h_data.data() + half,
                                gpu_n * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream));

        mul2_kernel<<<blocks, threads, 0, stream>>>(d_part, gpu_n);
        CUDA_OK(cudaGetLastError());

        CUDA_OK(cudaMemcpyAsync(h_data.data() + half,
                                d_part,
                                gpu_n * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                stream));

        // CPU часть
        #pragma omp parallel
        {
            #pragma omp single
            {
                cpu_mul2_omp(h_data.data(), cpu_n);
            }
        }

        // Ждём пока GPU закончит
        CUDA_OK(cudaStreamSynchronize(stream));

        auto t1 = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "N = " << N << "\n";
        std::cout << "First element (CPU part) = " << h_data[0] << "\n";
        std::cout << "First element of GPU part = " << h_data[half] << "\n";
        std::cout << "Hybrid total time (ms) = " << total_ms << "\n\n";

        CUDA_OK(cudaStreamDestroy(stream));
        CUDA_OK(cudaFree(d_part));
    }

    return 0;
}
