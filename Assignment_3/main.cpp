%%writefile ass3_task2.cu
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

__global__ void vec_add(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

float timeVecAdd(const float* dA, const float* dB, float* dC, int n,
                 int block, int iters)
{
    int grid = (n + block - 1) / block;

    vec_add<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        vec_add<<<grid, block>>>(dA, dB, dC, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters; // avg per launch
}

int main() {
    const int N = 1'000'000;
    const int iters = 300;

    // block sizes to test (>= 3 as required)
    const int blocks[] = {128, 256, 512, 1024};
    const int numTests = sizeof(blocks) / sizeof(blocks[0]);

    std::vector<float> hA(N), hB(N), hC(N), hRef(N);

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        hA[i] = dist(gen);
        hB[i] = dist(gen);
        hRef[i] = hA[i] + hB[i];
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Assignment 3 - Task 2 (CUDA): vector add, block size impact\n";
    std::cout << "N = " << N << ", iters = " << iters << "\n\n";

    std::cout << std::left
              << std::setw(10) << "Block"
              << std::setw(10) << "Grid"
              << std::setw(20) << "Avg kernel (ms)"
              << "\n";
    std::cout << std::string(40, '-') << "\n";

    float bestMs = 1e9f;
    int bestBlock = -1;

    for (int t = 0; t < numTests; ++t) {
        int block = blocks[t];
        int grid  = (N + block - 1) / block;

        float avgMs = timeVecAdd(dA, dB, dC, N, block, iters);

        std::cout << std::left
                  << std::setw(10) << block
                  << std::setw(10) << grid
                  << std::setw(20) << std::fixed << std::setprecision(6) << avgMs
                  << "\n";

        if (avgMs < bestMs) {
            bestMs = avgMs;
            bestBlock = block;
        }
    }

    // verify result using best block by running once more, then copy back
    int bestGrid = (N + bestBlock - 1) / bestBlock;
    vec_add<<<bestGrid, bestBlock>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost));

    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = std::fabs(hC[i] - hRef[i]);
        if (err > maxErr) maxErr = err;
    }

    const float EPS = 1e-4f;
    bool ok = (maxErr <= EPS);

    std::cout << "\nBest block size = " << bestBlock
              << " (avg " << bestMs << " ms)\n";
    std::cout << "Max abs error = " << maxErr << "\n";
    std::cout << "Correct = " << (ok ? "YES" : "NO") << "\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
