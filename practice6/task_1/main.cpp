#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>

static const char* KERNEL_SRC = R"CLC(
__kernel void vector_add(__global const int* A,
                         __global const int* B,
                         __global int* C) {
    int id = get_global_id(0);
    C[id] = A[id] + B[id];
}
)CLC";

static void check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::cerr << msg << " (err=" << err << ")\n";
        std::exit(1);
    }
}

static cl_device_id pick_device(cl_device_type dtype) {
    cl_uint pcount = 0;
    check(clGetPlatformIDs(0, nullptr, &pcount), "No OpenCL platforms");
    std::vector<cl_platform_id> plats(pcount);
    check(clGetPlatformIDs(pcount, plats.data(), nullptr), "clGetPlatformIDs failed");

    for (auto p : plats) {
        cl_uint dcount = 0;
        cl_int err = clGetDeviceIDs(p, dtype, 0, nullptr, &dcount);
        if (err != CL_SUCCESS || dcount == 0) continue;

        std::vector<cl_device_id> devs(dcount);
        check(clGetDeviceIDs(p, dtype, dcount, devs.data(), nullptr), "clGetDeviceIDs failed");
        return devs[0];
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

int main(int argc, char** argv) {
    // режим: cpu / gpu
    std::string mode = (argc >= 2) ? argv[1] : "gpu";
    cl_device_type dtype = (mode == "cpu") ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU;

    const size_t N = 10'000'000;

    // данные (1..100)
    std::vector<int> A(N), B(N), C(N), Cref(N);
    std::mt19937 gen(123);
    std::uniform_int_distribution<int> dist(1, 100);

    for (size_t i = 0; i < N; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
        Cref[i] = A[i] + B[i];
    }

    // устройство
    cl_device_id dev = pick_device(dtype);
    if (!dev) {
        std::cerr << "Device not found for mode: " << mode << "\n";
        std::cerr << "Попробуй: ./Ass3_1 cpu\n";
        return 1;
    }

    cl_int err = CL_SUCCESS;

    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    check(err, "clCreateContext failed");

    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    check(err, "clCreateCommandQueue failed");

    size_t src_len = std::strlen(KERNEL_SRC);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &KERNEL_SRC, &src_len, &err);
    check(err, "clCreateProgramWithSource failed");

    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logsz = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
        std::string log(logsz, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logsz, log.data(), nullptr);
        std::cerr << "Build log:\n" << log << "\n";
        check(err, "clBuildProgram failed");
    }

    cl_kernel krn = clCreateKernel(prog, "vector_add", &err);
    check(err, "clCreateKernel failed");

    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              N * sizeof(int), A.data(), &err);
    check(err, "clCreateBuffer A failed");

    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              N * sizeof(int), B.data(), &err);
    check(err, "clCreateBuffer B failed");

    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                              N * sizeof(int), nullptr, &err);
    check(err, "clCreateBuffer C failed");

    // аргументы
    check(clSetKernelArg(krn, 0, sizeof(cl_mem), &dA), "clSetKernelArg 0 failed");
    check(clSetKernelArg(krn, 1, sizeof(cl_mem), &dB), "clSetKernelArg 1 failed");
    check(clSetKernelArg(krn, 2, sizeof(cl_mem), &dC), "clSetKernelArg 2 failed");

    // замер полного времени (с memcpy)
    auto t_total1 = std::chrono::high_resolution_clock::now();

    // запуск ядра
    size_t global = N;
    cl_event evt{};
    check(clEnqueueNDRangeKernel(q, krn, 1, nullptr, &global, nullptr, 0, nullptr, &evt),
          "clEnqueueNDRangeKernel failed");
    clWaitForEvents(1, &evt);

    // чтение результата
    check(clEnqueueReadBuffer(q, dC, CL_TRUE, 0, N * sizeof(int), C.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer failed");

    auto t_total2 = std::chrono::high_resolution_clock::now();
    long long total_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_total2 - t_total1).count();

    // время ядра
    cl_ulong t0=0, t1=0;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t0, nullptr);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &t1, nullptr);
    double kernel_ms = (double)(t1 - t0) * 1e-6;

    // проверка
    int bad = 0;
    for (size_t i = 0; i < N; i++) {
        if (C[i] != Cref[i]) { bad = 1; break; }
    }

    std::cout << "Task 1 (OpenCL): Vector Add\n";
    std::cout << "Mode = " << mode << "\n";
    std::cout << "Device = " << dev_name(dev) << "\n";
    std::cout << "N = " << N << "\n";
    std::cout << "Kernel time (ms) = " << kernel_ms << "\n";
    std::cout << "Total time (ms) = " << total_ms << "\n";
    std::cout << "Correct = " << (bad ? "NO" : "YES") << "\n";

    // очистка
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
