#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <iomanip>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N = 1'000'000;
    if (argc >= 2) {
        N = std::stoll(argv[1]);
        if (N < 0) N = 0;
    }

    std::vector<double> global;
    std::vector<int> counts(size), displs(size);

    if (rank == 0) {
        global.resize((size_t)N);

        // Генерим данные только на rank 0
        std::mt19937_64 rng(123);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (long long i = 0; i < N; ++i) {
            global[(size_t)i] = dist(rng);
        }

        // Разбиение с остатком: первые r процессов получают на 1 элемент больше
        long long base = N / size;
        long long rem  = N % size;

        int offset = 0;
        for (int p = 0; p < size; ++p) {
            long long cnt = base + (p < rem ? 1 : 0);
            counts[p] = (int)cnt;
            displs[p] = offset;
            offset += counts[p];
        }
    }

    // Чтобы каждый процесс знал свой размер, разошлём counts
    int local_n = 0;
    MPI_Scatter(counts.data(), 1, MPI_INT, &local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> local((size_t)local_n);

    // Scatterv: каждый процесс получает свою часть
    MPI_Scatterv(
        rank == 0 ? global.data() : nullptr,
        rank == 0 ? counts.data() : nullptr,
        rank == 0 ? displs.data() : nullptr,
        MPI_DOUBLE,
        local.data(),
        local_n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Локальные суммы
    double local_sum = 0.0;
    double local_sum_sq = 0.0;

    for (int i = 0; i < local_n; ++i) {
        double x = local[(size_t)i];
        local_sum += x;
        local_sum_sq += x * x;
    }

    // Сбор на rank 0
    double global_sum = 0.0;
    double global_sum_sq = 0.0;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double mean = (N > 0) ? (global_sum / (double)N) : 0.0;

        // дисперсия = E[x^2] - (E[x])^2
        double ex2 = (N > 0) ? (global_sum_sq / (double)N) : 0.0;
        double var = ex2 - mean * mean;

        // из-за округления иногда var может стать чуть меньше 0
        if (var < 0.0) var = 0.0;

        double stddev = std::sqrt(var);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "N = " << N << "\n";
        std::cout << "MPI processes = " << size << "\n";
        std::cout << "Mean = " << mean << "\n";
        std::cout << "StdDev = " << stddev << "\n";
    }

    MPI_Finalize();
    return 0;
}
