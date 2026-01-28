#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

#define CHECK(call) \
if ((call) != cudaSuccess) { \
    std::cerr << "CUDA error\n"; \
    exit(1); \
}

// 1. Коалесцированный доступ
__global__ void coalesced_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = in[idx] * 2.0f;
}

// 2. Некоалесцированный доступ
__global__ void noncoalesced_kernel(const float* in, float* out, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx < n)
        out[idx] = in[idx] * 2.0f;
}

// 3. Shared memory
__global__ void shared_kernel(const float* in, float* out, int n) {
    __shared__ float tile[256];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (gid < n)
        tile[tid] = in[gid];
    __syncthreads();

    if (gid < n)
        out[gid] = tile[tid] * 2.0f;
}

// Замер времени
float measure(void(*kernel)(const float*, float*, int),
              const float* d_in, float* d_out,
              int n, dim3 grid, dim3 block)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    kernel<<<grid, block>>>(d_in, d_out, n);
    cudaEventRecord(start);
    kernel<<<grid, block>>>(d_in, d_out, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main() {
    const int N = 1'000'000;
    const int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK);

    std::vector<float> h_in(N, 1.0f), h_out(N);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    float t1 = measure(coalesced_kernel, d_in, d_out, N, grid, block);
    float t2 = measure(noncoalesced_kernel, d_in, d_out, N, grid, block);
    float t3 = measure(shared_kernel, d_in, d_out, N, grid, block);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Coalesced access time (ms)    = " << t1 << "\n";
    std::cout << "Non-coalesced access time (ms)= " << t2 << "\n";
    std::cout << "Shared memory time (ms)       = " << t3 << "\n";

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
