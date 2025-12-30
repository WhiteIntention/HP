#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <climits>
#include <cmath>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

using clk = std::chrono::high_resolution_clock;

struct GpuTime {
  float kernel_ms = 0.0f;     // время CUDA kernels (cudaEvent)
  long long total_ms = 0;     // общее время с memcpy (chrono)
  bool ok = false;            // Sorted = YES/NO
};

// ---- helpers ----
static inline bool is_sorted_ok(const std::vector<int>& a) {
  for (size_t i = 1; i < a.size(); i++)
    if (a[i - 1] > a[i]) return false;
  return true;
}

static inline std::vector<int> gen_data(int n, int seed = 123) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(1, 1000000);
  std::vector<int> a(n);
  for (int i = 0; i < n; i++) a[i] = dist(gen);
  return a;
}

// CPU SORTS (SEQUENTIAL)

// CPU MergeSort (bottom-up)
static void cpu_merge_pass(const std::vector<int>& src, std::vector<int>& dst, int l, int m, int r) {
  int i = l, j = m, k = l;
  while (i < m && j < r) dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
  while (i < m) dst[k++] = src[i++];
  while (j < r) dst[k++] = src[j++];
}

static void cpu_merge_sort(std::vector<int>& a) {
  int n = (int)a.size();
  std::vector<int> tmp(n);
  for (int w = 1; w < n; w *= 2) {
    for (int l = 0; l < n; l += 2 * w) {
      int m = std::min(n, l + w);
      int r = std::min(n, l + 2 * w);
      cpu_merge_pass(a, tmp, l, m, r);
    }
    a.swap(tmp);
  }
}

// CPU QuickSort (iterative, lomuto)
static int partition_lomuto(std::vector<int>& a, int l, int r) { // [l,r)
  int pivot = a[r - 1];
  int i = l;
  for (int j = l; j < r - 1; j++) {
    if (a[j] < pivot) {
      std::swap(a[i], a[j]);
      i++;
    }
  }
  std::swap(a[i], a[r - 1]);
  return i;
}

static void cpu_quick_sort(std::vector<int>& a) {
  std::vector<std::pair<int, int>> st;
  st.push_back({0, (int)a.size()});
  while (!st.empty()) {
    auto [l, r] = st.back();
    st.pop_back();
    if (r - l <= 1) continue;
    int p = partition_lomuto(a, l, r);
    st.push_back({l, p});
    st.push_back({p + 1, r});
  }
}

// CPU HeapSort
static void cpu_sift_down(std::vector<int>& a, int n, int i) {
  while (true) {
    int l = 2 * i + 1, r = 2 * i + 2;
    int best = i;
    if (l < n && a[l] > a[best]) best = l;
    if (r < n && a[r] > a[best]) best = r;
    if (best == i) break;
    std::swap(a[i], a[best]);
    i = best;
  }
}

static void cpu_heap_sort(std::vector<int>& a) {
  int n = (int)a.size();
  for (int i = n / 2 - 1; i >= 0; i--) cpu_sift_down(a, n, i);
  for (int last = n - 1; last > 0; last--) {
    std::swap(a[0], a[last]);
    cpu_sift_down(a, last, 0);
  }
}

template <class F>
static long long cpu_time_ms(F func, std::vector<int> a, bool& ok) {
  auto t1 = clk::now();
  func(a);
  auto t2 = clk::now();
  ok = is_sorted_ok(a);
  return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
}

// GPU MERGE SORT
__device__ void insertion_sort_dev(int* a, int n) {
  for (int i = 1; i < n; i++) {
    int key = a[i];
    int j = i - 1;
    while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
    a[j + 1] = key;
  }
}

__global__ void sort_chunks(const int* in, int* out, int n, int chunk) {
  int bid = (int)blockIdx.x;
  int start = bid * chunk;
  int len = chunk;
  if (start + len > n) len = n - start;
  if (len <= 0) return;

  extern __shared__ int s[];
  for (int i = (int)threadIdx.x; i < len; i += (int)blockDim.x) s[i] = in[start + i];
  __syncthreads();

  if (threadIdx.x == 0) insertion_sort_dev(s, len);
  __syncthreads();

  for (int i = (int)threadIdx.x; i < len; i += (int)blockDim.x) out[start + i] = s[i];
}

__device__ __forceinline__ int merge_path(const int* A, int m, const int* B, int n, int diag) {
  int lo = max(0, diag - n);
  int hi = min(diag, m);
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    int ai = mid;
    int bi = diag - mid;

    int aL = (ai > 0) ? A[ai - 1] : INT_MIN;
    int aR = (ai < m) ? A[ai] : INT_MAX;
    int bL = (bi > 0) ? B[bi - 1] : INT_MIN;
    int bR = (bi < n) ? B[bi] : INT_MAX;

    if (aL > bR) hi = mid;
    else if (bL > aR) lo = mid + 1;
    else return mid;
  }
  return lo;
}

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

  int d0 = (total * (int)threadIdx.x) / T;
  int d1 = (total * ((int)threadIdx.x + 1)) / T;

  int a_i0 = merge_path(A, m, B, k, d0);
  int b_i0 = d0 - a_i0;

  int outPos = start + d0;
  while (d0 < d1) {
    int aVal = (a_i0 < m) ? A[a_i0] : INT_MAX;
    int bVal = (b_i0 < k) ? B[b_i0] : INT_MAX;
    if (aVal <= bVal) { out[outPos++] = aVal; a_i0++; }
    else { out[outPos++] = bVal; b_i0++; }
    d0++;
  }
}

static GpuTime gpu_merge_sort_run(const std::vector<int>& h) {
  GpuTime t{};
  int N = (int)h.size();

  int *d_in=nullptr, *d_a=nullptr, *d_b=nullptr;
  cudaEvent_t e0{}, e1{};

  auto FAIL = [&](const char* msg) -> GpuTime {
    std::cerr << msg << "\n";
    if (d_in) cudaFree(d_in);
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    t.ok = false;
    return t;
  };

  if (cudaMalloc(&d_in, N*sizeof(int)) != cudaSuccess) return FAIL("cudaMalloc d_in failed");
  if (cudaMalloc(&d_a,  N*sizeof(int)) != cudaSuccess) return FAIL("cudaMalloc d_a failed");
  if (cudaMalloc(&d_b,  N*sizeof(int)) != cudaSuccess) return FAIL("cudaMalloc d_b failed");
  if (cudaEventCreate(&e0) != cudaSuccess) return FAIL("cudaEventCreate e0 failed");
  if (cudaEventCreate(&e1) != cudaSuccess) return FAIL("cudaEventCreate e1 failed");

  auto host_t1 = clk::now();
  if (cudaMemcpy(d_in, h.data(), N*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    return FAIL("cudaMemcpy H2D failed");

  cudaEventRecord(e0);

  const int CHUNK = 256;
  const int THREADS = 256;

  int blocks_chunks = (N + CHUNK - 1) / CHUNK;
  size_t shmem = CHUNK * sizeof(int);
  sort_chunks<<<blocks_chunks, THREADS, shmem>>>(d_in, d_a, N, CHUNK);
  if (cudaGetLastError() != cudaSuccess) return FAIL("kernel sort_chunks failed");

  int run = CHUNK;
  bool ping = true;
  while (run < N) {
    int pairs = (N + (2*run) - 1) / (2*run);
    if (ping) merge_pass<<<pairs, THREADS>>>(d_a, d_b, N, run);
    else      merge_pass<<<pairs, THREADS>>>(d_b, d_a, N, run);
    if (cudaGetLastError() != cudaSuccess) return FAIL("kernel merge_pass failed");
    ping = !ping;
    run *= 2;
  }

  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  cudaEventElapsedTime(&t.kernel_ms, e0, e1);

  std::vector<int> out(N);
  int* finalPtr = ping ? d_a : d_b;
  if (cudaMemcpy(out.data(), finalPtr, N*sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    return FAIL("cudaMemcpy D2H failed");

  auto host_t2 = clk::now();
  t.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(host_t2 - host_t1).count();
  t.ok = is_sorted_ok(out);

  cudaFree(d_in); cudaFree(d_a); cudaFree(d_b);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return t;
}

// GPU QUICK SORT (parallel partition + thrust for small)
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

  int leftPos  = prefix[i];
  int rightPos = i - prefix[i];

  int outIdx = isLeft ? (l + leftPos) : (l + leftCount + rightPos);
  out[outIdx] = v;
}

struct Seg { int l, r; };

static GpuTime gpu_quick_sort_run(const std::vector<int>& h) {
  GpuTime t{};
  int N = (int)h.size();

  int *d_main=nullptr, *d_tmp=nullptr;
  int *d_flags=nullptr, *d_prefix=nullptr;
  void* d_scan_tmp=nullptr;
  size_t scan_bytes=0;
  cudaEvent_t e0{}, e1{};

  auto FAIL = [&](const char* msg) -> GpuTime {
    std::cerr << msg << "\n";
    if (d_main) cudaFree(d_main);
    if (d_tmp) cudaFree(d_tmp);
    if (d_flags) cudaFree(d_flags);
    if (d_prefix) cudaFree(d_prefix);
    if (d_scan_tmp) cudaFree(d_scan_tmp);
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    t.ok = false;
    return t;
  };

  if (cudaMalloc(&d_main, N*sizeof(int)) != cudaSuccess) return FAIL("cudaMalloc d_main failed");
  if (cudaMalloc(&d_tmp,  N*sizeof(int)) != cudaSuccess) return FAIL("cudaMalloc d_tmp failed");
  if (cudaMalloc(&d_flags,  N*sizeof(int)) != cudaSuccess) return FAIL("cudaMalloc d_flags failed");
  if (cudaMalloc(&d_prefix, N*sizeof(int)) != cudaSuccess) return FAIL("cudaMalloc d_prefix failed");
  if (cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, d_flags, d_prefix, N) != cudaSuccess)
    return FAIL("CUB scan size query failed");
  if (cudaMalloc(&d_scan_tmp, scan_bytes) != cudaSuccess) return FAIL("cudaMalloc d_scan_tmp failed");

  if (cudaEventCreate(&e0) != cudaSuccess) return FAIL("cudaEventCreate e0 failed");
  if (cudaEventCreate(&e1) != cudaSuccess) return FAIL("cudaEventCreate e1 failed");

  auto host_t1 = clk::now();
  if (cudaMemcpy(d_main, h.data(), N*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    return FAIL("cudaMemcpy H2D failed");

  cudaEventRecord(e0);

  const int SMALL = 8192;
  const int THREADS = 256;

  std::vector<Seg> st;
  st.push_back({0, N});

  while (!st.empty()) {
    Seg s = st.back(); st.pop_back();
    int len = s.r - s.l;
    if (len <= 1) continue;

    if (len <= SMALL) {
      thrust::device_ptr<int> p = thrust::device_pointer_cast(d_main);
      thrust::sort(p + s.l, p + s.r);
      continue;
    }

    int pivot = 0;
    int mid = s.l + len / 2;
    if (cudaMemcpy(&pivot, d_main + mid, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
      return FAIL("cudaMemcpy pivot failed");

    int blocks = (len + THREADS - 1) / THREADS;

    make_flags<<<blocks, THREADS>>>(d_main, d_flags, s.l, s.r, pivot);
    if (cudaGetLastError() != cudaSuccess) return FAIL("kernel make_flags failed");

    if (cub::DeviceScan::ExclusiveSum(d_scan_tmp, scan_bytes, d_flags, d_prefix, len) != cudaSuccess)
      return FAIL("CUB scan failed");

    int lastFlag=0, lastPref=0;
    if (cudaMemcpy(&lastFlag, d_flags + (len - 1), sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
      return FAIL("cudaMemcpy lastFlag failed");
    if (cudaMemcpy(&lastPref, d_prefix + (len - 1), sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
      return FAIL("cudaMemcpy lastPref failed");

    int leftCount = lastPref + lastFlag;

    scatter_partition<<<blocks, THREADS>>>(d_main, d_tmp, d_flags, d_prefix, s.l, s.r, leftCount);
    if (cudaGetLastError() != cudaSuccess) return FAIL("kernel scatter failed");

    if (cudaMemcpy(d_main + s.l, d_tmp + s.l, len*sizeof(int), cudaMemcpyDeviceToDevice) != cudaSuccess)
      return FAIL("cudaMemcpy D2D segment failed");

    int m = s.l + leftCount;
    st.push_back({s.l, m});
    st.push_back({m, s.r});
  }

  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  cudaEventElapsedTime(&t.kernel_ms, e0, e1);

  std::vector<int> out(N);
  if (cudaMemcpy(out.data(), d_main, N*sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    return FAIL("cudaMemcpy D2H failed");

  auto host_t2 = clk::now();
  t.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(host_t2 - host_t1).count();
  t.ok = is_sorted_ok(out);

  cudaFree(d_main); cudaFree(d_tmp);
  cudaFree(d_flags); cudaFree(d_prefix);
  cudaFree(d_scan_tmp);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return t;
}

// GPU HEAP SORT (heapify parallel + extract in one kernel)
__device__ void sift_down_dev(int* a, int n, int i) {
  while (true) {
    int l = 2*i + 1;
    int r = 2*i + 2;
    int best = i;
    if (l < n && a[l] > a[best]) best = l;
    if (r < n && a[r] > a[best]) best = r;
    if (best == i) break;
    int tmp = a[i]; a[i] = a[best]; a[best] = tmp;
    i = best;
  }
}

__global__ void heapify_level(int* a, int n, int start, int end) {
  int idx = start + (int)blockIdx.x*(int)blockDim.x + (int)threadIdx.x;
  if (idx > end) return;
  sift_down_dev(a, n, idx);
}

__global__ void heapsort_extract_all(int* a, int n) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int last = n - 1; last > 0; last--) {
    int tmp = a[0]; a[0] = a[last]; a[last] = tmp;
    sift_down_dev(a, last, 0);
  }
}

static void build_heap_levels(int* d_a, int n) {
  int last_parent = (n/2) - 1;
  if (last_parent < 0) return;

  int maxDepth = (int)std::floor(std::log2((double)(last_parent + 1)));
  const int THREADS = 256;

  for (int d = maxDepth; d >= 0; d--) {
    int start = (1 << d) - 1;
    int end   = (1 << (d+1)) - 2;
    if (start > last_parent) continue;
    if (end > last_parent) end = last_parent;

    int count = end - start + 1;
    int blocks = (count + THREADS - 1) / THREADS;
    heapify_level<<<blocks, THREADS>>>(d_a, n, start, end);
    cudaDeviceSynchronize();
  }
}

static GpuTime gpu_heap_sort_run(const std::vector<int>& h) {
  GpuTime t{};
  int N = (int)h.size();

  int* d_a=nullptr;
  cudaEvent_t e0{}, e1{};

  auto FAIL = [&](const char* msg) -> GpuTime {
    std::cerr << msg << "\n";
    if (d_a) cudaFree(d_a);
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    t.ok = false;
    return t;
  };

  if (cudaMalloc(&d_a, N*sizeof(int)) != cudaSuccess) return FAIL("cudaMalloc d_a failed");
  if (cudaEventCreate(&e0) != cudaSuccess) return FAIL("cudaEventCreate e0 failed");
  if (cudaEventCreate(&e1) != cudaSuccess) return FAIL("cudaEventCreate e1 failed");

  auto host_t1 = clk::now();
  if (cudaMemcpy(d_a, h.data(), N*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    return FAIL("cudaMemcpy H2D failed");

  cudaEventRecord(e0);

  build_heap_levels(d_a, N);
  heapsort_extract_all<<<1,1>>>(d_a, N);
  if (cudaGetLastError() != cudaSuccess) return FAIL("kernel heapsort_extract_all failed");
  cudaDeviceSynchronize();

  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  cudaEventElapsedTime(&t.kernel_ms, e0, e1);

  std::vector<int> out(N);
  if (cudaMemcpy(out.data(), d_a, N*sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    return FAIL("cudaMemcpy D2H failed");

  auto host_t2 = clk::now();
  t.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(host_t2 - host_t1).count();
  t.ok = is_sorted_ok(out);

  cudaFree(d_a);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return t;
}

// MAIN BENCH 
int main() {
  std::vector<int> sizes = {10000, 100000, 1000000};

  std::cout << "CPU vs GPU Sorting Benchmark (Merge / Quick / Heap)\n\n";

  for (int N : sizes) {
    auto base = gen_data(N, 123);

    std::cout << "N = " << N << "\n";

    // CPU
    bool ok1=false, ok2=false, ok3=false;
    long long cpu_ms_merge = cpu_time_ms(cpu_merge_sort, base, ok1);
    long long cpu_ms_quick = cpu_time_ms(cpu_quick_sort, base, ok2);
    long long cpu_ms_heap  = cpu_time_ms(cpu_heap_sort,  base, ok3);

    // GPU
    auto g_merge = gpu_merge_sort_run(base);
    auto g_quick = gpu_quick_sort_run(base);
    auto g_heap  = gpu_heap_sort_run(base);

    std::cout << "CPU MergeSort: " << cpu_ms_merge << " ms, OK=" << (ok1?"YES":"NO") << "\n";
    std::cout << "GPU MergeSort: kernel=" << g_merge.kernel_ms << " ms, total=" << g_merge.total_ms
              << " ms, OK=" << (g_merge.ok?"YES":"NO") << "\n";

    std::cout << "CPU QuickSort: " << cpu_ms_quick << " ms, OK=" << (ok2?"YES":"NO") << "\n";
    std::cout << "GPU QuickSort: kernel=" << g_quick.kernel_ms << " ms, total=" << g_quick.total_ms
              << " ms, OK=" << (g_quick.ok?"YES":"NO") << "\n";

    std::cout << "CPU HeapSort : " << cpu_ms_heap << " ms, OK=" << (ok3?"YES":"NO") << "\n";
    std::cout << "GPU HeapSort : kernel=" << g_heap.kernel_ms << " ms, total=" << g_heap.total_ms
              << " ms, OK=" << (g_heap.ok?"YES":"NO") << "\n";

    std::cout << "----------------------------------------\n\n";
  }

  return 0;
}
