#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << "\n"; \
        return 1; \
    } \
} while(0)

// простая сортировка для блока (insertion sort)
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

// каждый блок сортирует свой кусок массива
__global__ void sort_chunks_kernel(const int* in, int* out, int n, int chunk) {
    int blockId = blockIdx.x;
    int start = blockId * chunk;
    int len = chunk;
    if (start + len > n) len = n - start;
    if (len <= 0) return;

    extern __shared__ int s[];

    // грузим в shared
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        s[i] = in[start + i];
    }
    __syncthreads();

    // сортируем в одном потоке (для 10k/100k и chunk=256 это ок)
    if (threadIdx.x == 0) {
        insertion_sort(s, len);
    }
    __syncthreads();

    // выгружаем назад
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
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

        int a_left  = (a_i > 0) ? A[a_i - 1] : INT_MIN;
        int a_right = (a_i < m) ? A[a_i]     : INT_MAX;
        int b_left  = (b_i > 0) ? B[b_i - 1] : INT_MIN;
        int b_right = (b_i < n) ? B[b_i]     : INT_MAX;

        if (a_left > b_right) hi = mid;
        else if (b_left > a_right) lo = mid + 1;
        else return mid;
    }
    return lo;
}

// параллельное слияние двух отсортированных подмассивов
__global__ void merge_pass_kernel(const int* in, int* out, int n, int run) {
    int pairId = blockIdx.x;
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
    int T = blockDim.x;

    int diag0 = (total * threadIdx.x) / T;
    int diag1 = (total * (threadIdx.x + 1)) / T;

    int a_idx0 = merge_path(A, m, B, k, diag0);
    int b_idx0 = diag0 - a_idx0;

    int a_idx1 = merge_path(A, m, B, k, diag1);
    int b_idx1 = diag1 - a_idx1;

    int outPos = start + diag0;

    // сливаем свой диапазон [diag0..diag1)
    while (diag0 < diag1) {
        int a_val = (a_idx0 < m) ? A[a_idx0] : INT_MAX;
        int b_val = (b_idx0 < k) ? B[b_idx0] : INT_MAX;

        if (a_val <= b_val) {
            out[outPos++] = a_val;
            a_idx0++;
        } else {
            out[outPos++] = b_val;
            b_idx0++;
        }
        diag0++;
    }
}

bool check_sorted(const std::vector<int>& v) {
    for (size_t i = 1; i < v.size(); i++) {
        if (v[i - 1] > v[i]) return false;
    }
    return true;
}

int run_case(int N) {
    const int CHUNK = 256;               // размер подмассива на блок
    const int THREADS = 256;

    std::vector<int> h_in(N);
    std::mt19937 gen(123);
    std::uniform_int_distribution<int> dist(1, 1000000);
    for (int i = 0; i < N; i++) h_in[i] = dist(gen);

    int *d_in = nullptr, *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_a,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b,  N * sizeof(int)));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // время: весь GPU sort + memcpy (как “общее” время)
    auto host_t1 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(e0));

    // 1) сортируем куски
    int blocks_chunks = (N + CHUNK - 1) / CHUNK;
    size_t shmem = CHUNK * sizeof(int);
    sort_chunks_kernel<<<blocks_chunks, THREADS, shmem>>>(d_in, d_a, N, CHUNK);
    CUDA_CHECK(cudaGetLastError());

    // 2) параллельные merge-проходы (run удваивается)
    int run = CHUNK;
    bool ping = true; // true: in=d_a, out=d_b
    while (run < N) {
        int pairs = (N + (2 * run) - 1) / (2 * run);
        if (ping) {
            merge_pass_kernel<<<pairs, THREADS>>>(d_a, d_b, N, run);
        } else {
            merge_pass_kernel<<<pairs, THREADS>>>(d_b, d_a, N, run);
        }
        CUDA_CHECK(cudaGetLastError());
        ping = !ping;
        run *= 2;
    }

    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));

    // результат лежит в (ping ? d_a : d_b) потому что ping переключили
    std::vector<int> h_out(N);
    if (ping) {
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_a, N * sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_b, N * sizeof(int), cudaMemcpyDeviceToHost));
    }

    auto host_t2 = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(host_t2 - host_t1).count();

    std::cout << "N = " << N << "\n";
    std::cout << "GPU kernels time (ms) = " << gpu_ms << "\n";
    std::cout << "Total time incl memcpy (ms) = " << total_ms << "\n";
    std::cout << "Sorted = " << (check_sorted(h_out) ? "YES" : "NO") << "\n\n";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    return 0;
}

int main() {
    std::cout << "Task 4: Merge sort on GPU (CUDA)\n\n";

    // 10 000 и 100 000 по условию
    if (run_case(10000) != 0) return 1;
    if (run_case(100000) != 0) return 1;

    return 0;
}
