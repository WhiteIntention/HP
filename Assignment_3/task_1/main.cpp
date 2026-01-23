%%writefile ass3_task1.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
                  << " at " << __FILE__ << ":" << __LINE__         \
                  << std::endl;                                    \
        std::exit(1);                                              \
    }                                                              \
} while(0)

// Kernels 

// 1) Only global memory
__global__ void mul_global(const float* __restrict__ in,
                           float* __restrict__ out,
                           float k,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * k;
}

// 2) Using shared memory
__global__ void mul_shared(const float* __restrict__ in,
                           float* __restrict__ out,
                           float k,
                           int n)
{
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) s[tid] = in[idx];
    __syncthreads();

    if (idx < n) s[tid] = s[tid] * k;
    __syncthreads();

    if (idx < n) out[idx] = s[tid];
}

// Timing helper

template <typename LaunchFunc>
float timeKernel(LaunchFunc launch, int iters) {
    // warm-up
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch();
        CUDA_CHECK(cudaGetLastError()); // <-- КЛЮЧЕВО: ловим ошибки запуска
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

int main() {
    const int N = 1'000'000;
    const float k = 3.5f;

    std::vector<float> h_in(N), h_out(N), h_ref(N);

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) h_in[i] = dist(gen);
    for (int i = 0; i < N; ++i) h_ref[i] = h_in[i] * k;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

    const int block = 256;
    const int grid  = (N + block - 1) / block;
    const int iters = 200;
    const size_t shmem = (size_t)block * sizeof(float);

    // Global timing
    float global_ms = timeKernel([&](){
        mul_global<<<grid, block>>>(d_in, d_out, k, N);
    }, iters);

    // Shared timing
    float shared_ms = timeKernel([&](){
        mul_shared<<<grid, block, shmem>>>(d_in, d_out, k, N);
    }, iters);

    // Copy back after shared (last executed)
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify with tolerance + max error
    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = std::fabs(h_out[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    const float EPS = 1e-4f;
    bool ok = (max_err <= EPS);

    std::cout << "Assignment 3 - Task 1 (CUDA): element-wise multiply\n";
    std::cout << "N = " << N << ", k = " << k << "\n";
    std::cout << "Block = " << block << ", Grid = " << grid << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Global memory kernel avg time (ms) = " << global_ms << "\n";
    std::cout << "Shared memory kernel avg time (ms) = " << shared_ms << "\n";
    std::cout << "Max abs error = " << max_err << "\n";
    std::cout << "Correct = " << (ok ? "YES" : "NO") << "\n";
    std::cout << "Speed ratio (shared/global) = " << (shared_ms / global_ms) << "\n";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
