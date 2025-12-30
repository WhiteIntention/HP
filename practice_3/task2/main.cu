#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define CUDA_OK(x) do { \
  cudaError_t e = (x); \
  if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << " at line " << __LINE__ << "\n"; \
    return 1; \
  } \
} while(0)

struct Seg { int l, r; }; // [l, r)

__global__ void make_flags(const int* in, int* flags, int l, int r, int pivot) {
  int idx = l + (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (idx >= r) return;
  flags[idx - l] = (in[idx] < pivot) ? 1 : 0;
}

__global__ void scatter_partition(const int* in, int* out,
                                  const int* flags, const int* prefix,
                                  int l, int r, int leftCount) {
  int idx = l + (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (idx >= r) return;

  int i = idx - l;
  int v = in[idx];
  int isLeft = flags[i];

  // prefix = exclusive scan по flags
  int leftPos  = prefix[i];
  int rightPos = i - prefix[i];

  int outIdx = isLeft ? (l + leftPos) : (l + leftCount + rightPos);
  out[outIdx] = v;
}

__global__ void insertion_sort_range(int* a, int l, int r) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int i = l + 1; i < r; i++) {
    int key = a[i];
    int j = i - 1;
    while (j >= l && a[j] > key) {
      a[j + 1] = a[j];
      j--;
    }
    a[j + 1] = key;
  }
}

bool check_sorted(const std::vector<int>& v) {
  for (size_t i = 1; i < v.size(); i++)
    if (v[i - 1] > v[i]) return false;
  return true;
}

int main() {
  const int N = 100000;      // один массив (можешь менять)
  const int SMALL = 8192;    // маленькие сегменты добиваем insertion-sort

  // --- данные на CPU ---
  std::vector<int> h(N);
  std::mt19937 gen(123);
  std::uniform_int_distribution<int> dist(1, 1000000);
  for (int i = 0; i < N; i++) h[i] = dist(gen);

  // --- память GPU ---
  int *d_main = nullptr, *d_tmp = nullptr;
  CUDA_OK(cudaMalloc(&d_main, N * sizeof(int)));
  CUDA_OK(cudaMalloc(&d_tmp,  N * sizeof(int)));
  CUDA_OK(cudaMemcpy(d_main, h.data(), N * sizeof(int), cudaMemcpyHostToDevice));

  // flags/prefix (максимум N)
  int *d_flags = nullptr, *d_prefix = nullptr;
  CUDA_OK(cudaMalloc(&d_flags,  N * sizeof(int)));
  CUDA_OK(cudaMalloc(&d_prefix, N * sizeof(int)));

  // temp для CUB scan
  void* d_scan_tmp = nullptr;
  size_t scan_bytes = 0;
  CUDA_OK(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, d_flags, d_prefix, N));
  CUDA_OK(cudaMalloc(&d_scan_tmp, scan_bytes));

  cudaEvent_t e0, e1;
  CUDA_OK(cudaEventCreate(&e0));
  CUDA_OK(cudaEventCreate(&e1));
  CUDA_OK(cudaEventRecord(e0));

  // стек сегментов на CPU
  std::vector<Seg> st;
  st.push_back({0, N});

  while (!st.empty()) {
    Seg s = st.back();
    st.pop_back();

    int len = s.r - s.l;
    if (len <= 1) continue;

    if (len <= SMALL) {
      thrust::device_ptr<int> p = thrust::device_pointer_cast(d_main);
      thrust::sort(p + s.l, p + s.r);
      continue;
    }


    // pivot: читаем 1 элемент (это нормально для учебной версии)
    int pivot = 0;
    int mid = s.l + len / 2;
    CUDA_OK(cudaMemcpy(&pivot, d_main + mid, sizeof(int), cudaMemcpyDeviceToHost));

    int threads = 256;
    int blocks  = (len + threads - 1) / threads;

    // flags для сегмента пишем в начало буфера flags[0..len)
    make_flags<<<blocks, threads>>>(d_main, d_flags, s.l, s.r, pivot);
    CUDA_OK(cudaGetLastError());

    // prefix sum по flags (len элементов)
    CUDA_OK(cub::DeviceScan::ExclusiveSum(d_scan_tmp, scan_bytes, d_flags, d_prefix, len));

    // leftCount = sum(flags) = prefix[last] + flags[last]
    int lastFlag = 0, lastPref = 0;
    CUDA_OK(cudaMemcpy(&lastFlag, d_flags + (len - 1), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&lastPref, d_prefix + (len - 1), sizeof(int), cudaMemcpyDeviceToHost));
    int leftCount = lastPref + lastFlag;

    // scatter в d_tmp на позиции [l..r)
    scatter_partition<<<blocks, threads>>>(d_main, d_tmp, d_flags, d_prefix, s.l, s.r, leftCount);
    CUDA_OK(cudaGetLastError());

    // копируем назад только сегмент (важно для корректности)
    CUDA_OK(cudaMemcpy(d_main + s.l, d_tmp + s.l, len * sizeof(int), cudaMemcpyDeviceToDevice));

    // две части (левая < pivot, правая >= pivot)
    int m = s.l + leftCount;
    st.push_back({s.l, m});
    st.push_back({m, s.r});
  }

  CUDA_OK(cudaEventRecord(e1));
  CUDA_OK(cudaEventSynchronize(e1));
  float gpu_ms = 0.0f;
  CUDA_OK(cudaEventElapsedTime(&gpu_ms, e0, e1));

  // результат на CPU
  std::vector<int> out(N);
  CUDA_OK(cudaMemcpy(out.data(), d_main, N * sizeof(int), cudaMemcpyDeviceToHost));

  std::cout << "Task: Parallel QuickSort on CUDA (parallel partition)\n\n";
  std::cout << "N = " << N << "\n";
  std::cout << "GPU time (ms) = " << gpu_ms << "\n";
  std::cout << "Sorted = " << (check_sorted(out) ? "YES" : "NO") << "\n";

  CUDA_OK(cudaFree(d_main));
  CUDA_OK(cudaFree(d_tmp));
  CUDA_OK(cudaFree(d_flags));
  CUDA_OK(cudaFree(d_prefix));
  CUDA_OK(cudaFree(d_scan_tmp));
  CUDA_OK(cudaEventDestroy(e0));
  CUDA_OK(cudaEventDestroy(e1));

  return 0;
}

