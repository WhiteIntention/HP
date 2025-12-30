#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_OK_RET(x) do { \
  cudaError_t e = (x); \
  if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << " at line " << __LINE__ << "\n"; \
    return 1; \
  } \
} while(0)

#define CUDA_OK_VOID(x) do { \
  cudaError_t e = (x); \
  if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << " at line " << __LINE__ << "\n"; \
    return; \
  } \
} while(0)

// sift-down для max-heap
__device__ void sift_down(int* a, int n, int i) {
  while (true) {
    int l = 2 * i + 1;
    int r = 2 * i + 2;
    int best = i;

    if (l < n && a[l] > a[best]) best = l;
    if (r < n && a[r] > a[best]) best = r;

    if (best == i) break;

    int tmp = a[i];
    a[i] = a[best];
    a[best] = tmp;

    i = best;
  }
}

// heapify: просеивание узлов уровня [start..end] (все узлы одного уровня)
__global__ void heapify_level(int* a, int n, int start, int end) {
  int idx = start + (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (idx > end) return;
  sift_down(a, n, idx);
}

// извлечение элементов (последовательно, но внутри ОДНОГО kernel)
__global__ void heapsort_extract_all(int* a, int n) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  for (int last = n - 1; last > 0; last--) {
    int tmp = a[0];
    a[0] = a[last];
    a[last] = tmp;

    sift_down(a, last, 0);
  }
}

bool is_sorted_cpu(const std::vector<int>& v) {
  for (size_t i = 1; i < v.size(); i++)
    if (v[i - 1] > v[i]) return false;
  return true;
}

// построение кучи: параллельно по уровням снизу вверх
void build_heap_parallel(int* d_a, int n) {
  int last_parent = (n / 2) - 1;
  if (last_parent < 0) return;

  // глубина последнего внутреннего узла
  int maxDepth = (int)std::floor(std::log2((double)(last_parent + 1)));

  const int threads = 256;

  for (int d = maxDepth; d >= 0; d--) {
    int start = (1 << d) - 1;
    int end = (1 << (d + 1)) - 2;
    if (start > last_parent) continue;
    if (end > last_parent) end = last_parent;

    int count = end - start + 1;
    int blocks = (count + threads - 1) / threads;

    heapify_level<<<blocks, threads>>>(d_a, n, start, end);
    CUDA_OK_VOID(cudaGetLastError());
    CUDA_OK_VOID(cudaDeviceSynchronize());
  }
}

int main() {
  const int N = 100000; // один массив

  // CPU данные
  std::vector<int> h(N);
  std::mt19937 gen(123);
  std::uniform_int_distribution<int> dist(1, 1000000);
  for (int i = 0; i < N; i++) h[i] = dist(gen);

  // GPU память
  int* d_a = nullptr;
  CUDA_OK_RET(cudaMalloc(&d_a, N * sizeof(int)));
  CUDA_OK_RET(cudaMemcpy(d_a, h.data(), N * sizeof(int), cudaMemcpyHostToDevice));

  // таймер GPU
  cudaEvent_t e0, e1;
  CUDA_OK_RET(cudaEventCreate(&e0));
  CUDA_OK_RET(cudaEventCreate(&e1));
  CUDA_OK_RET(cudaEventRecord(e0));

  // 1) heapify (параллельно где возможно)
  build_heap_parallel(d_a, N);

  // 2) извлечение (в одном kernel)
  heapsort_extract_all<<<1, 1>>>(d_a, N);
  CUDA_OK_RET(cudaGetLastError());
  CUDA_OK_RET(cudaDeviceSynchronize());

  CUDA_OK_RET(cudaEventRecord(e1));
  CUDA_OK_RET(cudaEventSynchronize(e1));

  float gpu_ms = 0.0f;
  CUDA_OK_RET(cudaEventElapsedTime(&gpu_ms, e0, e1));

  // результат
  std::vector<int> out(N);
  CUDA_OK_RET(cudaMemcpy(out.data(), d_a, N * sizeof(int), cudaMemcpyDeviceToHost));

  std::cout << "Task: Parallel Heap Sort on CUDA\n\n";
  std::cout << "N = " << N << "\n";
  std::cout << "GPU time (ms) = " << gpu_ms << "\n";
  std::cout << "Sorted = " << (is_sorted_cpu(out) ? "YES" : "NO") << "\n";

  CUDA_OK_RET(cudaFree(d_a));
  CUDA_OK_RET(cudaEventDestroy(e0));
  CUDA_OK_RET(cudaEventDestroy(e1));

  return 0;
}
