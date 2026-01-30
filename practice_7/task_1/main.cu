#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

#define CUDA_OK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        std::exit(1); \
    } \
} while (0)

__global__ void reduceSumShared(const float* __restrict__ data,
                                float* __restrict__ partial,
                                int n)
{
    extern __shared__ float sh[];
    const unsigned tid = threadIdx.x;

    const unsigned base = blockIdx.x * (blockDim.x * 2u) + tid;

    float local = 0.0f;
    if (base < (unsigned)n) local += data[base];
    if (base + blockDim.x < (unsigned)n) local += data[base + blockDim.x];

    sh[tid] = local;
    __syncthreads();

    for (unsigned step = blockDim.x / 2u; step > 0; step >>= 1u) {
        if (tid < step) sh[tid] += sh[tid + step];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sh[0];
}

static double cpuReferenceSum(const std::vector<float>& a)
{
    double s = 0.0;
    for (float v : a) s += static_cast<double>(v);
    return s;
}

static double runGpuReduction(const std::vector<float>& host, float& ms_out)
{
    const int n = (int)host.size();
    const int threads = 256;

    float* d_in = nullptr;
    CUDA_OK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_OK(cudaMemcpy(d_in, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int blocks0 = (n + (threads * 2 - 1)) / (threads * 2);

    float* d_bufA = nullptr;
    float* d_bufB = nullptr;
    CUDA_OK(cudaMalloc(&d_bufA, blocks0 * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_bufB, blocks0 * sizeof(float)));

    const float* curIn = d_in;
    float* curOut = d_bufA;
    int curN = n;

    cudaEvent_t evStart, evStop;
    CUDA_OK(cudaEventCreate(&evStart));
    CUDA_OK(cudaEventCreate(&evStop));
    CUDA_OK(cudaEventRecord(evStart));

    bool toggle = false;
    while (true) {
        int blocks = (curN + (threads * 2 - 1)) / (threads * 2);

        reduceSumShared<<<blocks, threads, threads * sizeof(float)>>>(curIn, curOut, curN);
        CUDA_OK(cudaGetLastError());

        if (blocks == 1) break;

        curN = blocks;
        curIn = curOut;

        toggle = !toggle;
        curOut = toggle ? d_bufB : d_bufA;
    }

    CUDA_OK(cudaEventRecord(evStop));
    CUDA_OK(cudaEventSynchronize(evStop));
    CUDA_OK(cudaEventElapsedTime(&ms_out, evStart, evStop));

    float gpu_f = 0.0f;
    CUDA_OK(cudaMemcpy(&gpu_f, curOut, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_OK(cudaEventDestroy(evStart));
    CUDA_OK(cudaEventDestroy(evStop));

    CUDA_OK(cudaFree(d_in));
    CUDA_OK(cudaFree(d_bufA));
    CUDA_OK(cudaFree(d_bufB));

    return static_cast<double>(gpu_f);
}

int main()
{
    const std::vector<int> tests = { 1024, 1'000'000, 10'000'000 };

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::cout << std::fixed << std::setprecision(6);

    for (int N : tests) {
        std::vector<float> x(N);

        for (int i = 0; i < N; ++i) x[i] = dist(rng);

        const double cpu = cpuReferenceSum(x);

        float gpu_ms = 0.0f;
        const double gpu = runGpuReduction(x, gpu_ms);

        const double abs_err = std::fabs(cpu - gpu);
        const double rel_err = abs_err / (std::fabs(cpu) + 1e-12);

        std::cout << "N = " << N << "\n";
        std::cout << "CPU sum = " << cpu << "\n";
        std::cout << "GPU sum = " << gpu << "\n";
        std::cout << "Abs error = " << abs_err << "\n";
        std::cout << "Rel error = " << rel_err << "\n";
        std::cout << "GPU time (ms) = " << gpu_ms << "\n";
        std::cout << "\n";
    }

    return 0;
}
