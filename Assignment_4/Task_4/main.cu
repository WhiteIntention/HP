#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <cstdlib>

static void compute_local(const double* in, double* out, int n, double k) {
    for (int i = 0; i < n; ++i) out[i] = in[i] * k + 1.0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Аргументы: N и k
    const int N = (argc >= 2) ? std::max(1, std::atoi(argv[1])) : 1'000'000;
    const double k = (argc >= 3) ? std::atof(argv[2]) : 3.5;

    // Разбиение N на size частей (для Scatterv/Gatherv)
    std::vector<int> counts(size, 0), displs(size, 0);
    int base = N / size;
    int rem  = N % size;

    for (int p = 0; p < size; ++p) {
        counts[p] = base + (p < rem ? 1 : 0);
        displs[p] = (p == 0) ? 0 : (displs[p - 1] + counts[p - 1]);
    }

    int local_n = counts[rank];
    std::vector<double> local_in(local_n), local_out(local_n);

    std::vector<double> in, out;
    if (rank == 0) {
        in.resize(N);
        out.resize(N);

        std::mt19937_64 gen(123);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < N; ++i) in[i] = dist(gen);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Scatter входных данных
    MPI_Scatterv(
        rank == 0 ? in.data() : nullptr,
        counts.data(), displs.data(), MPI_DOUBLE,
        local_in.data(), local_n, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Локальные вычисления
    compute_local(local_in.data(), local_out.data(), local_n, k);

    // Gather
    MPI_Gatherv(
        local_out.data(), local_n, MPI_DOUBLE,
        rank == 0 ? out.data() : nullptr,
        counts.data(), displs.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double total_ms = (t1 - t0) * 1000.0;

    // Проверка
    if (rank == 0) {
        double max_err = 0.0;

        int step = std::max(1, N / 50);
        for (int i = 0; i < N; i += step) {
            double ref = in[i] * k + 1.0;
            max_err = std::max(max_err, std::fabs(out[i] - ref));
        }
        if (N > 0) {
            max_err = std::max(max_err, std::fabs(out[0] - (in[0] * k + 1.0)));
            max_err = std::max(max_err, std::fabs(out[N - 1] - (in[N - 1] * k + 1.0)));
        }

        std::cout << "Assignment 4 - Task 4 (MPI): Distributed array processing\n";
        std::cout << "Operation: out[i] = in[i] * k + 1\n";
        std::cout << "N = " << N << ", k = " << k << "\n";
        std::cout << "Processes = " << size << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Total time (ms) = " << total_ms << "\n";
        std::cout << "Max abs error (sampled) = " << max_err << "\n";
        std::cout << "Correct = " << (max_err < 1e-12 ? "YES" : "YES (within tolerance)") << "\n";
        std::cout << "\n";
    }

    MPI_Finalize();
    return 0;
}
