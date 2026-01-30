#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
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

// Ядро: умножаем каждый элемент на 2
__global__ void mul2_kernel(float* data, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) data[gid] *= 2.0f;
}

int main()
{
    const std::vector<int> sizes = { 1'000'000, 10'000'000 };

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::cout << std::fixed << std::setprecision(6);

    for (int N : sizes) {
        std::vector<float> h_data(N);

        // генерим входные данные на CPU
        for (int i = 0; i < N; ++i) {
            h_data[i] = dist(rng);
        }

        float* d_data = nullptr;
        CUDA_OK(cudaMalloc(&d_data, N * sizeof(float)));

        // чтобы было понятно, что именно мы меряем:
        // отдельно H2D, отдельно kernel, отдельно D2H
        float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f;

        // H2D
        {
            cudaEvent_t s, e;
            CUDA_OK(cudaEventCreate(&s));
            CUDA_OK(cudaEventCreate(&e));
            CUDA_OK(cudaEventRecord(s));
            CUDA_OK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_OK(cudaEventRecord(e));
            CUDA_OK(cudaEventSynchronize(e));
            CUDA_OK(cudaEventElapsedTime(&h2d_ms, s, e));
            CUDA_OK(cudaEventDestroy(s));
            CUDA_OK(cudaEventDestroy(e));
        }

        // kernel
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        {
            cudaEvent_t s, e;
            CUDA_OK(cudaEventCreate(&s));
            CUDA_OK(cudaEventCreate(&e));
            CUDA_OK(cudaEventRecord(s));
            mul2_kernel<<<blocks, threads>>>(d_data, N);
            CUDA_OK(cudaGetLastError());
            CUDA_OK(cudaEventRecord(e));
            CUDA_OK(cudaEventSynchronize(e));
            CUDA_OK(cudaEventElapsedTime(&kernel_ms, s, e));
            CUDA_OK(cudaEventDestroy(s));
            CUDA_OK(cudaEventDestroy(e));
        }

        // D2H
        {
            cudaEvent_t s, e;
            CUDA_OK(cudaEventCreate(&s));
            CUDA_OK(cudaEventCreate(&e));
            CUDA_OK(cudaEventRecord(s));
            CUDA_OK(cudaMemcpy(h_data.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaEventRecord(e));
            CUDA_OK(cudaEventSynchronize(e));
            CUDA_OK(cudaEventElapsedTime(&d2h_ms, s, e));
            CUDA_OK(cudaEventDestroy(s));
            CUDA_OK(cudaEventDestroy(e));
        }

        // проверка, что реально умножили
        std::cout << "N = " << N << "\n";
        std::cout << "First element = " << h_data[0] << "\n";
        std::cout << "GPU H2D time (ms) = " << h2d_ms << "\n";
        std::cout << "GPU kernel time (ms) = " << kernel_ms << "\n";
        std::cout << "GPU D2H time (ms) = " << d2h_ms << "\n";
        std::cout << "GPU total time (ms) = " << (h2d_ms + kernel_ms + d2h_ms) << "\n\n";

        CUDA_OK(cudaFree(d_data));
    }

    return 0;
}
