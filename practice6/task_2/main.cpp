#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>

static const char* KERNEL_SRC = R"CLC(
__kernel void matmul(__global const float* A,
                     __global const float* B,
                     __global float* C,
                     int N, int M, int K) {
    int row = get_global_id(0); // 0..N-1
    int col = get_global_id(1); // 0..K-1
    if (row >= N || col >= K) return;

    float sum = 0.0f;
    for (int i = 0; i < M; i++) {
        sum += A[row * M + i] * B[i * K + col];
    }
    C[row * K + col] = sum;
}
)CLC";

static void die(const char* msg, cl_int err) {
    std::cerr << msg << " (err=" << err << ")\n";
    std::exit(1);
}

static cl_device_id pick_device(cl_device_type dtype) {
    cl_uint pcount = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &pcount);
    if (err != CL_SUCCESS || pcount == 0) return nullptr;

    std::vector<cl_platform_id> plats(pcount);
    err = clGetPlatformIDs(pcount, plats.data(), nullptr);
    if (err != CL_SUCCESS) return nullptr;

    for (auto p : plats) {
        cl_uint dcount = 0;
        err = clGetDeviceIDs(p, dtype, 0, nullptr, &dcount);
        if (err != CL_SUCCESS || dcount == 0) continue;

        std::vector<cl_device_id> devs(dcount);
        err = clGetDeviceIDs(p, dtype, dcount, devs.data(), nullptr);
        if (err == CL_SUCCESS && !devs.empty()) return devs[0];
    }
    return nullptr;
}

static std::string dev_name(cl_device_id d) {
    size_t sz = 0;
    clGetDeviceInfo(d, CL_DEVICE_NAME, 0, nullptr, &sz);
    std::string s(sz, '\0');
    clGetDeviceInfo(d, CL_DEVICE_NAME, sz, s.data(), nullptr);
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

static void cpu_matmul(const std::vector<float>& A,
                       const std::vector<float>& B,
                       std::vector<float>& C,
                       int N, int M, int K) {
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < K; c++) {
            float sum = 0.0f;
            for (int i = 0; i < M; i++) {
                sum += A[r*M + i] * B[i*K + c];
            }
            C[r*K + c] = sum;
        }
    }
}

int main(int argc, char** argv) {
    // размеры
    int N = (argc >= 2) ? std::stoi(argv[1]) : 256;
    int M = (argc >= 3) ? std::stoi(argv[2]) : 256;
    int K = (argc >= 4) ? std::stoi(argv[3]) : 256;

    std::string mode = (argc >= 5) ? argv[4] : "gpu";
    cl_device_type dtype = (mode == "cpu") ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU;

    // матрицы
    std::vector<float> A((size_t)N*M), B((size_t)M*K), Cgpu((size_t)N*K), Ccpu((size_t)N*K);

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    // CPU time
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    cpu_matmul(A, B, Ccpu, N, M, K);
    auto cpu_t2 = std::chrono::high_resolution_clock::now();
    long long cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_t2 - cpu_t1).count();

    cl_device_id dev = pick_device(dtype);
    if (!dev) {
        std::cerr << "OpenCL device not found for mode: " << mode << "\n";
        return 1;
    }

    cl_int err = CL_SUCCESS;

    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) die("clCreateContext failed", err);

    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) die("clCreateCommandQueue failed", err);

    size_t src_len = std::strlen(KERNEL_SRC);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &KERNEL_SRC, &src_len, &err);
    if (err != CL_SUCCESS) die("clCreateProgramWithSource failed", err);

    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logsz = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
        std::string log(logsz, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logsz, log.data(), nullptr);
        std::cerr << "Build log:\n" << log << "\n";
        die("clBuildProgram failed", err);
    }

    cl_kernel krn = clCreateKernel(prog, "matmul", &err);
    if (err != CL_SUCCESS) die("clCreateKernel failed", err);

    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              (size_t)N*M*sizeof(float), (void*)A.data(), &err);
    if (err != CL_SUCCESS) die("clCreateBuffer A failed", err);

    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              (size_t)M*K*sizeof(float), (void*)B.data(), &err);
    if (err != CL_SUCCESS) die("clCreateBuffer B failed", err);

    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                              (size_t)N*K*sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) die("clCreateBuffer C failed", err);

    // аргументы (включая N,M,K)
    err  = clSetKernelArg(krn, 0, sizeof(cl_mem), &dA);
    err |= clSetKernelArg(krn, 1, sizeof(cl_mem), &dB);
    err |= clSetKernelArg(krn, 2, sizeof(cl_mem), &dC);
    err |= clSetKernelArg(krn, 3, sizeof(int), &N);
    err |= clSetKernelArg(krn, 4, sizeof(int), &M);
    err |= clSetKernelArg(krn, 5, sizeof(int), &K);
    if (err != CL_SUCCESS) die("clSetKernelArg failed", err);

    // global size = (N, K)
    size_t global[2] = { (size_t)N, (size_t)K };

    // total time
    auto gpu_total1 = std::chrono::high_resolution_clock::now();

    cl_event evt{};
    err = clEnqueueNDRangeKernel(q, krn, 2, nullptr, global, nullptr, 0, nullptr, &evt);
    if (err != CL_SUCCESS) die("clEnqueueNDRangeKernel failed", err);

    clWaitForEvents(1, &evt);

    err = clEnqueueReadBuffer(q, dC, CL_TRUE, 0, (size_t)N*K*sizeof(float), Cgpu.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) die("clEnqueueReadBuffer failed", err);

    auto gpu_total2 = std::chrono::high_resolution_clock::now();
    long long gpu_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_total2 - gpu_total1).count();

    // kernel time
    cl_ulong t0=0, t1=0;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t0, nullptr);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &t1, nullptr);
    double kernel_ms = (double)(t1 - t0) * 1e-6;

    // проверка
    double max_err = 0.0;
    for (size_t i = 0; i < (size_t)N*K; i++) {
        max_err = std::max(max_err, (double)std::fabs(Cgpu[i] - Ccpu[i]));
    }
    bool ok = (max_err < 1e-3);

    std::cout << "Task 2 (OpenCL): Matrix Multiplication\n";
    std::cout << "Mode = " << mode << "\n";
    std::cout << "Device = " << dev_name(dev) << "\n";
    std::cout << "N=" << N << " M=" << M << " K=" << K << "\n\n";

    std::cout << "CPU time (ms) = " << cpu_ms << "\n";
    std::cout << "GPU kernel time (ms) = " << kernel_ms << "\n";
    std::cout << "GPU total time (ms) = " << gpu_total_ms << "\n";
    std::cout << "Max error = " << max_err << "\n";
    std::cout << "Correct = " << (ok ? "YES" : "NO") << "\n";

    // cleanup
    clReleaseEvent(evt);
    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
    clReleaseKernel(krn);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    return 0;
}
