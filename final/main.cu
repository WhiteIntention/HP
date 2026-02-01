#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <omp.h>

// FIR-фильтр как каузальная 1D-свёртка: y[i] = sum_{j=0..K-1} h[j] * x[i-j], если i-j < 0 -> 0.
// Делаем 3 реализации: CPU (seq), CPU (OpenMP), GPU (CUDA). Сравниваем время и max abs error.

#define CUDA_CHECK(call) do {                              \
    cudaError_t err = (call);                               \
    if (err != cudaSuccess) {                               \
        std::cerr << "CUDA error: "                         \
                  << cudaGetErrorString(err)                \
                  << " (" << __FILE__ << ":" << __LINE__    \
                  << ")\n";                                 \
        std::exit(1);                                       \
    }                                                       \
} while (0)

// Фильтр h одинаковый для всех потоков, поэтому удобно хранить его в constant memory.
// Максимальная длина фильтра для хранения в constant memory
static constexpr int MAX_K = 1024;
__constant__ float c_h[MAX_K];

// чтобы сравнение с GPU было более честным (на GPU float и другой порядок суммирования).
// CPU FIR (эталон): аккумулируем в double для более стабильного сравнения
static void fir_cpu_seq(const std::vector<float>& x,
                        const std::vector<float>& h,
                        std::vector<float>& y)
{
    const int N = (int)x.size();
    const int K = (int)h.size();
    y.assign(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        double acc = 0.0;
        for (int j = 0; j < K; ++j) {
            int idx = i - j;
            if (idx >= 0) acc += (double)h[(size_t)j] * (double)x[(size_t)idx];
        }
        y[(size_t)i] = (float)acc;
    }
}

// OpenMP: параллелим по i (по выходным элементам), потому что y[i] независимы.
// Каждый поток считает свой acc локально, гонок данных нет.
static void fir_cpu_omp(const std::vector<float>& x,
                        const std::vector<float>& h,
                        std::vector<float>& y)
{
    const int N = (int)x.size();
    const int K = (int)h.size();
    y.assign(N, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double acc = 0.0;
        for (int j = 0; j < K; ++j) {
            int idx = i - j;
            if (idx >= 0) acc += (double)h[(size_t)j] * (double)x[(size_t)idx];
        }
        y[(size_t)i] = (float)acc;
    }
}

// GPU kernel: один поток -> один y[i]
// Фильтр из constant memory (c_h), сигнал x из global memory.
__global__ void fir_kernel(const float* __restrict__ x,
                           float* __restrict__ y,
                           int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float acc = 0.0f;
    // c_h лежит в constant memory, хорошо подходит для "одинаковых" чтений многими потоками
    for (int j = 0; j < K; ++j) {
        int idx = i - j;
        if (idx >= 0) acc += c_h[j] * x[idx];
    }
    y[i] = acc;
}

static double max_abs_error(const std::vector<float>& a, const std::vector<float>& b)
{
    double m = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        double e = std::fabs((double)a[i] - (double)b[i]);
        if (e > m) m = e;
    }
    return m;
}

static double ms_since(const std::chrono::high_resolution_clock::time_point& t0,
                       const std::chrono::high_resolution_clock::time_point& t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main()
{
    // Наборы размеров для экспериментов: меняем длину сигнала N и длину фильтра K.
    // По ним сравниваем, где CPU/OMP/GPU выигрывают.
    std::vector<int> Ns = { 1'000'000, 10'000'000, 50'000'000 };
    std::vector<int> Ks = { 15, 31, 63, 127, 255 };

    // Чтобы не падать, ограничим K под MAX_K
    Ks.erase(std::remove_if(Ks.begin(), Ks.end(),
                            [](int k){ return k <= 0 || k > MAX_K; }),
             Ks.end());

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "FIR assignment: y[i] = sum_{j=0..K-1} h[j] * x[i-j], x<0 -> 0\n";
    std::cout << "OpenMP threads = " << omp_get_max_threads() << "\n\n";

    // Буферы на GPU выделяю один раз под максимальный N, чтобы не делать cudaMalloc в каждом эксперименте.
    float* d_x = nullptr;
    float* d_y = nullptr;
    size_t maxN = (size_t)*std::max_element(Ns.begin(), Ns.end());
    CUDA_CHECK(cudaMalloc(&d_x, maxN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, maxN * sizeof(float)));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    for (int K : Ks) {
        // Фильтр (h) фиксируем на каждом K
        std::vector<float> h((size_t)K);
        for (int j = 0; j < K; ++j) h[(size_t)j] = dist(rng) * 0.1f; // небольшие веса

        // Кладем коэффициенты фильтра в constant memory: часто читается всеми потоками,
        // поэтому так обычно быстрее, чем хранить h в global.
        CUDA_CHECK(cudaMemcpyToSymbol(c_h, h.data(), (size_t)K * sizeof(float)));

        for (int N : Ns) {
            std::vector<float> x((size_t)N);
            for (int i = 0; i < N; ++i) x[(size_t)i] = dist(rng);

            std::vector<float> y_ref, y_cpu_omp, y_gpu;

            // CPU sequential time
            auto t0 = std::chrono::high_resolution_clock::now();
            fir_cpu_seq(x, h, y_ref);
            auto t1 = std::chrono::high_resolution_clock::now();
            double cpu_seq_ms = ms_since(t0, t1);

            // CPU OpenMP time
            auto t2 = std::chrono::high_resolution_clock::now();
            fir_cpu_omp(x, h, y_cpu_omp);
            auto t3 = std::chrono::high_resolution_clock::now();
            double cpu_omp_ms = ms_since(t2, t3);

            // GPU: замеряем H2D, kernel, D2H и total
            float h2d_ms = 0.0f, k_ms = 0.0f, d2h_ms = 0.0f, total_ms = 0.0f;

            // Для GPU отдельно копирование Host->Device, время ядра и Device->Host.
            // Важны и kernel-time (чистые вычисления), и total-time (реальная стоимость с копированием).
            // H2D
            CUDA_CHECK(cudaEventRecord(ev_start));
            CUDA_CHECK(cudaMemcpy(d_x, x.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, ev_start, ev_stop));

            // kernel
            int threads = 256;
            int blocks = (N + threads - 1) / threads;

            CUDA_CHECK(cudaEventRecord(ev_start));
            fir_kernel<<<blocks, threads>>>(d_x, d_y, N, K);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            CUDA_CHECK(cudaEventElapsedTime(&k_ms, ev_start, ev_stop));

            // D2H
            y_gpu.resize((size_t)N);
            CUDA_CHECK(cudaEventRecord(ev_start));
            CUDA_CHECK(cudaMemcpy(y_gpu.data(), d_y, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, ev_start, ev_stop));

            total_ms = h2d_ms + k_ms + d2h_ms;

            // Ошибка: max abs error относительно CPU seq (эталона). 
            // Малые расхождения допустимы из-за float и порядка сумм.
            double err_gpu = max_abs_error(y_ref, y_gpu);
            double err_omp = max_abs_error(y_ref, y_cpu_omp);

            // Результаты
            std::cout << "N=" << N << "  K=" << K << "\n";
            std::cout << "CPU seq (ms) = " << cpu_seq_ms << "\n";
            std::cout << "CPU omp (ms) = " << cpu_omp_ms << ", max abs err vs ref = " << err_omp << "\n";
            std::cout << "GPU H2D (ms) = " << h2d_ms << "\n";
            std::cout << "GPU kernel (ms) = " << k_ms << "\n";
            std::cout << "GPU D2H (ms) = " << d2h_ms << "\n";
            std::cout << "GPU total (ms) = " << total_ms << ", max abs err vs ref = " << err_gpu << "\n\n";
        }
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return 0;
}
