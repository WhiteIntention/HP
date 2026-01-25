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

// 1) Coalesced access: thread i -> element i
__global__ void kernel_coalesced(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = in[idx] * 2.0f;
}

// 2) Non-coalesced access: strided access pattern
// thread i -> element (i * stride) % n
__global__ void kernel_noncoalesced(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int n,
                                    int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int j = (idx * stride) % n;
        out[j] = in[j] * 2.0f;
    }
}

// Timing helper
template <typename KernelLaunch>
float timeKernel(KernelLaunch launch, int iters)
{
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch();
        CUDA_CHECK(cudaGetLastError());
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
    const int iters = 300;
    const int stride = 32;   // breaks coalescing

    std::vector<float> h_in(N), h_out(N), h_ref(N);

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        h_in[i] = dist(gen);
        h_ref[i] = h_in[i] * 2.0f;
    }

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid  = (N + block - 1) / block;

    // Coalesced
    float coalesced_ms = timeKernel([&]() {
        kernel_coalesced<<<grid, block>>>(d_in, d_out, N);
    }, iters);

    // Non-coalesced
    float noncoal_ms = timeKernel([&]() {
        kernel_noncoalesced<<<grid, block>>>(d_in, d_out, N, stride);
    }, iters);

    // Verify correctness (after coalesced run)
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = std::fabs(h_out[i] - h_ref[i]);
        if (err > maxErr) maxErr = err;
    }

    const float EPS = 1e-4f;
    bool ok = (maxErr <= EPS);

    std::cout << "Assignment 3 - Task 3 (CUDA): Coalesced vs Non-coalesced\n";
    std::cout << "N = " << N << ", Block = " << block
              << ", Grid = " << grid << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Coalesced access avg time (ms)      = "
              << coalesced_ms << "\n";
    std::cout << "Non-coalesced access avg time (ms) = "
              << noncoal_ms << "\n";
    std::cout << "Slowdown (non/coalesced) = "
              << (noncoal_ms / coalesced_ms) << "\n";
    std::cout << "Max abs error = " << maxErr << "\n";
    std::cout << "Correct = " << (ok ? "YES" : "NO") << "\n";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
