#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <climits>

#define CUDA_OK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) << " at line " << __LINE__ << "\n"; \
        return; \
    } \
} while(0)

#define CUDA_OK_RET(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) << " at line " << __LINE__ << "\n"; \
        return 1; \
    } \
} while(0)

// маленькая сортировка (для chunk) - insertion
__device__ void insertion_sort(int* a, int n) {
    for (int i = 1; i < n; i++) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

// 1) каждый CUDA-блок сортирует свой chunk
__global__ void sort_chunks(const int* in, int* out, int n, int chunk) {
    int bid = (int)blockIdx.x;
    int start = bid * chunk;
    int len = chunk;
    if (start + len > n) len = n - start;
    if (len <= 0) return;

    extern __shared__ int s[];

    // загрузка в shared
    for (int i = (int)threadIdx.x; i < len; i += (int)blockDim.x) {
        s[i] = in[start + i];
    }
    __syncthreads();

    // сортировка в одном потоке
    if (threadIdx.x == 0) {
        insertion_sort(s, len);
    }
    __syncthreads();

    // выгрузка назад
    for (int i = (int)threadIdx.x; i < len; i += (int)blockDim.x) {
        out[start + i] = s[i];
    }
}

// merge-path (поиск разреза по диагонали)
__device__ __forceinline__ int merge_path(const int* A, int m, const int* B, int n, int diag) {
    int lo = max(0, diag - n);
    int hi = min(diag, m);

    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int a_i = mid;
        int b_i = diag - mid;

        int aL = (a_i > 0) ? A[a_i - 1] : INT_MIN;
        int aR = (a_i < m) ? A[a_i] : INT_MAX;
        int bL = (b_i > 0) ? B[b_i - 1] : INT_MIN;
        int bR = (b_i < n) ? B[b_i] : INT_MAX;

        if (aL > bR) hi = mid;
        else if (bL > aR) lo = mid + 1;
        else return mid;
    }
    return lo;
}

// 2) один CUDA-блок сливает пару отсортированных кусков длины run
__global__ void merge_pass(const int* in, int* out, int n, int run) {
    int pairId = (int)blockIdx.x;
    int start = pairId * (2 * run);

    int a0 = start;
    int a1 = min(start + run, n);
    int b0 = a1;
    int b1 = min(start + 2 * run, n);

    int m = a1 - a0;
    int k = b1 - b0;
    if (m <= 0) return;

    const int* A = in + a0;
    const int* B = in + b0;

    int total = m + k;
    int T = (int)blockDim.x;

    int diag0 = (total * (int)threadIdx.x) / T;
    int diag1 = (total * ((int)threadIdx.x + 1)) / T;

    int a_i0 = merge_path(A, m, B, k, diag0);
    int b_i0 = diag0 - a_i0;

    int a_i1 = merge_path(A, m, B, k, diag1);
    int b_i1 = diag1 - a_i1;

    int outPos = start + diag0;

    // слияние своего диапазона
    while (diag0 < diag1) {
        int aVal = (a_i0 < m) ? A[a_i0] : INT_MAX;
        int bVal = (b_i0 < k) ? B[b_i0] : INT_MAX;

        if (aVal <= bVal) {
            out[outPos++] = aVal;
            a_i0++;
        } else {
            out[outPos++] = bVal;
            b_i0++;
        }
        diag0++;
    }
}

bool check_sorted(const std::vector<int>& v) {
    for (size_t i = 1; i < v.size(); i++)
        if (v[i - 1] > v[i]) return false;
    return true;
}

void gpu_merge_sort(int N) {
    const int CHUNK = 256;     // размер блока данных
    const int THREADS = 256;   // потоки в блоке

    // входные данные (CPU)
    std::vector<int> h_in(N);
    std::mt19937 gen(123);
    std::uniform_int_distribution<int> dist(1, 1000000);
    for (int i = 0; i < N; i++) h_in[i] = dist(gen);

    // память на GPU
    int *d_in = nullptr, *d_a = nullptr, *d_b = nullptr;
    CUDA_OK(cudaMalloc(&d_in, N * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_a,  N * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_b,  N * sizeof(int)));

    // события для времени GPU
    cudaEvent_t e0, e1;
    CUDA_OK(cudaEventCreate(&e0));
    CUDA_OK(cudaEventCreate(&e1));

    // общее время (включая memcpy)
    auto host_t1 = std::chrono::high_resolution_clock::now();

    CUDA_OK(cudaMemcpy(d_in, h_in.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_OK(cudaEventRecord(e0));

    // 1) сортируем chunks
    int blocks_chunks = (N + CHUNK - 1) / CHUNK;
    size_t shmem = CHUNK * sizeof(int);
    sort_chunks<<<blocks_chunks, THREADS, shmem>>>(d_in, d_a, N, CHUNK);
    CUDA_OK(cudaGetLastError());

    // 2) merge-проходы по парам
    int run = CHUNK;
    bool ping = true; // true: in=d_a, out=d_b
    while (run < N) {
        int pairs = (N + (2 * run) - 1) / (2 * run);
        if (ping) merge_pass<<<pairs, THREADS>>>(d_a, d_b, N, run);
        else      merge_pass<<<pairs, THREADS>>>(d_b, d_a, N, run);

        CUDA_OK(cudaGetLastError());
        ping = !ping;
        run *= 2;
    }

    CUDA_OK(cudaEventRecord(e1));
    CUDA_OK(cudaEventSynchronize(e1));

    float gpu_ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&gpu_ms, e0, e1));

    // результат
    std::vector<int> h_out(N);
    if (ping) CUDA_OK(cudaMemcpy(h_out.data(), d_a, N * sizeof(int), cudaMemcpyDeviceToHost));
    else      CUDA_OK(cudaMemcpy(h_out.data(), d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    auto host_t2 = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(host_t2 - host_t1).count();

    std::cout << "N = " << N << "\n";
    std::cout << "GPU kernels time (ms) = " << gpu_ms << "\n";
    std::cout << "Total time incl memcpy (ms) = " << total_ms << "\n";
    std::cout << "Sorted = " << (check_sorted(h_out) ? "YES" : "NO") << "\n\n";

    CUDA_OK(cudaFree(d_in));
    CUDA_OK(cudaFree(d_a));
    CUDA_OK(cudaFree(d_b));
    CUDA_OK(cudaEventDestroy(e0));
    CUDA_OK(cudaEventDestroy(e1));
}

int main() {
    std::cout << "CUDA merge sort (chunks + pairwise merge)\n\n";

    gpu_merge_sort(10000);
    gpu_merge_sort(100000);

    return 0;
}

