#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
                  << " at " << __FILE__ << ":" << __LINE__         \
                  << std::endl;                                    \
        std::exit(1);                                              \
    }                                                              \
} while(0)

// CUDA kernel 
// Sum using global memory + atomicAdd
__global__ void sum_global(const float* __restrict__ data,
                           float* result,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(result, data[idx]);
    }
}

int main() {
    const int N = 100'000;
    const int block = 256;
    const int grid  = (N + block - 1) / block;

    std::vector<float> h_data(N);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < N; ++i)
        h_data[i] = dist(gen);

    // CPU version 
    auto cpu_start = std::chrono::high_resolution_clock::now();

    float cpu_sum = 0.0f;
    for (int i = 0; i < N; ++i)
        cpu_sum += h_data[i];

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // CUDA version 
    float *d_data = nullptr;
    float *d_result = nullptr;

    CUDA_CHECK(cudaMalloc(&d_data,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(),
                          N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

    sum_global<<<grid, block>>>(d_data, d_result, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    sum_global<<<grid, block>>>(d_data, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    float gpu_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_result,
                          sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Output
    std::cout << "Assignment 4 - Task 1: Array Sum (CPU vs CUDA)\n";
    std::cout << "N = " << N << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CPU sum  = " << cpu_sum << "\n";
    std::cout << "GPU sum  = " << gpu_sum << "\n";
    std::cout << "CPU time (ms) = " << cpu_ms << "\n";
    std::cout << "GPU time (ms) = " << gpu_ms << "\n";

    std::cout << "Difference = " << std::abs(cpu_sum - gpu_sum) << "\n";

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
