#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <thread>
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

// GPU kernel: out[i] = in[i] * k
__global__ void mul_kernel(const float* __restrict__ in,
                           float* __restrict__ out,
                           float k,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * k;
}

// CPU function for a range [start, end)
void mul_cpu_range(const float* in, float* out, float k, int start, int end)
{
    for (int i = start; i < end; ++i) out[i] = in[i] * k;
}

int main() {
    const int N = 1'000'000;
    const float k = 3.5f;

    // Hybrid split: 50% CPU, 50% GPU (можно менять)
    const int split = N / 2; // [0..split) -> CPU, [split..N) -> GPU

    std::vector<float> h_in(N), h_cpu(N), h_gpu(N), h_hybrid(N), h_ref(N);

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        h_in[i]  = dist(gen);
        h_ref[i] = h_in[i] * k;
    }

    // CPU-only
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    mul_cpu_range(h_in.data(), h_cpu.data(), k, 0, N);
    auto cpu_t2 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t2 - cpu_t1).count();

    // GPU-only 
    float *d_in=nullptr, *d_out=nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid  = (N + block - 1) / block;

    // warm-up
    mul_kernel<<<grid, block>>>(d_in, d_out, k, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t gs, ge;
    CUDA_CHECK(cudaEventCreate(&gs));
    CUDA_CHECK(cudaEventCreate(&ge));

    CUDA_CHECK(cudaEventRecord(gs));
    mul_kernel<<<grid, block>>>(d_in, d_out, k, N);
    CUDA_CHECK(cudaEventRecord(ge));
    CUDA_CHECK(cudaEventSynchronize(ge));

    float gpu_kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_ms, gs, ge));

    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Hybrid (CPU + GPU одновременно) 
    // Для GPU-части используем отдельные буферы только на вторую половину
    int N2 = N - split;

    float *d_in2=nullptr, *d_out2=nullptr;
    CUDA_CHECK(cudaMalloc(&d_in2,  N2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out2, N2 * sizeof(float)));

    // start total hybrid timer
    auto hy_t1 = std::chrono::high_resolution_clock::now();

    // копируем на GPU вторую часть (host->device)
    CUDA_CHECK(cudaMemcpy(d_in2, h_in.data() + split, N2 * sizeof(float), cudaMemcpyHostToDevice));

    // запускаем CPU-часть в отдельном потоке
    std::thread cpu_thread([&](){
        mul_cpu_range(h_in.data(), h_hybrid.data(), k, 0, split);
    });

    // параллельно запускаем GPU-ядро на вторую часть
    int grid2 = (N2 + block - 1) / block;
    mul_kernel<<<grid2, block>>>(d_in2, d_out2, k, N2);
    CUDA_CHECK(cudaGetLastError());

    // ждём GPU
    CUDA_CHECK(cudaDeviceSynchronize());

    // копируем результат GPU-части обратно
    CUDA_CHECK(cudaMemcpy(h_hybrid.data() + split, d_out2, N2 * sizeof(float), cudaMemcpyDeviceToHost));

    // ждём CPU поток
    cpu_thread.join();

    auto hy_t2 = std::chrono::high_resolution_clock::now();
    double hybrid_ms = std::chrono::duration<double, std::milli>(hy_t2 - hy_t1).count();

    auto max_err = [&](const std::vector<float>& x){
        float m = 0.0f;
        for (int i = 0; i < N; ++i) {
            float e = std::fabs(x[i] - h_ref[i]);
            if (e > m) m = e;
        }
        return m;
    };

    float cpu_err = max_err(h_cpu);
    float gpu_err = max_err(h_gpu);
    float hy_err  = max_err(h_hybrid);

    // Output 
    std::cout << "Assignment 4 - Task 3: Hybrid CPU+GPU array processing\n";
    std::cout << "Operation: out[i] = in[i] * k\n";
    std::cout << "N = " << N << ", k = " << k << "\n";
    std::cout << "Split: CPU [0.." << split << "), GPU [" << split << ".." << N << ")\n";
    std::cout << "GPU block = " << block << "\n\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CPU-only time (ms)         = " << cpu_ms << " | max err = " << cpu_err << "\n";
    std::cout << "GPU-only kernel time (ms)  = " << gpu_kernel_ms << " | max err = " << gpu_err << "\n";
    std::cout << "Hybrid total time (ms)     = " << hybrid_ms << " | max err = " << hy_err << "\n";

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(gs));
    CUDA_CHECK(cudaEventDestroy(ge));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_in2));
    CUDA_CHECK(cudaFree(d_out2));

    return 0;
}
