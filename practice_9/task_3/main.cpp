#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>

static inline int idx(int i, int j, int N) {
    return i * N + j;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Размер графа задаём параметром: mpirun -np 4 ./9_3 64
    int N = 64;
    if (argc >= 2) {
        N = std::max(1, std::atoi(argv[1]));
    }

    // Scatter требует одинаковое число строк на процесс -> делаем padding
    int rows_per_proc = (N + size - 1) / size;
    int Npad = rows_per_proc * size;

    const int INF = 1'000'000'000;

    std::vector<int> global;
    if (rank == 0) {
        global.assign(Npad * N, INF);

        std::mt19937 rng(123);
        std::uniform_int_distribution<int> wdist(1, 20);
        std::uniform_real_distribution<double> edist(0.0, 1.0);

        // Генерация графа: диагональ 0, ребро с вероятностью ~0.25
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i == j) {
                    global[idx(i, j, N)] = 0;
                } else {
                    if (edist(rng) < 0.25) global[idx(i, j, N)] = wdist(rng);
                }
            }
        }

        // padded строки оставляем INF (они не участвуют, gi >= N)
        for (int i = N; i < Npad; ++i) {
            for (int j = 0; j < N; ++j) {
                global[idx(i, j, N)] = INF;
            }
        }
    }

    std::vector<int> local(rows_per_proc * N, INF);

    MPI_Scatter(
        rank == 0 ? global.data() : nullptr,
        rows_per_proc * N, MPI_INT,
        local.data(),
        rows_per_proc * N, MPI_INT,
        0, MPI_COMM_WORLD
    );
    // синхронизация перед замером
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Этот буфер будет содержать полную матрицу на каждой итерации
    // Мы собираем её через Allgather, чтобы у каждого процесса были актуальные строки
    std::vector<int> full(Npad * N, INF);

    // Инициализация full на первой итерации
    MPI_Allgather(
        local.data(), rows_per_proc * N, MPI_INT,
        full.data(),  rows_per_proc * N, MPI_INT,
        MPI_COMM_WORLD
    );

    // Алгоритм Флойда-Уоршелла
    // full содержит актуальные строки на начало итерации k
    // local обновляет только свои строки
    for (int k = 0; k < N; ++k) {
        // Чтобы не трогать padded строки, обновляем только gi < N
        for (int lr = 0; lr < rows_per_proc; ++lr) {
            int gi = rank * rows_per_proc + lr;
            if (gi >= N) continue;

            int dik = full[idx(gi, k, N)];
            if (dik >= INF) continue;

            for (int j = 0; j < N; ++j) {
                int dkj = full[idx(k, j, N)];
                if (dkj >= INF) continue;

                int cand = dik + dkj;
                int &dij = local[idx(lr, j, N)];
                if (cand < dij) dij = cand;
            }
        }

        // После обновления своей части делаем Allgather,
        // чтобы все процессы получили обновлённые строки на следующую итерацию
        MPI_Allgather(
            local.data(), rows_per_proc * N, MPI_INT,
            full.data(),  rows_per_proc * N, MPI_INT,
            MPI_COMM_WORLD
        );
    }

    // Сбор на rank 0 для финального вывода (можно и full использовать, но так понятнее)
    MPI_Gather(
        local.data(),
        rows_per_proc * N, MPI_INT,
        rank == 0 ? global.data() : nullptr,
        rows_per_proc * N, MPI_INT,
        0, MPI_COMM_WORLD
    );

    // синхронизация после вычислений
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "N = " << N << "\n";
        std::cout << "MPI processes = " << size << "\n";
            std::cout << "Execution time: "
              << (end_time - start_time)
              << " seconds\n";

        // Чтобы вывод не был огромным, печатаем матрицу только если N небольшое
        if (N <= 16) {
            std::cout << "Shortest paths matrix:\n";
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    int v = global[idx(i, j, N)];
                    if (v >= INF / 2) std::cout << "INF ";
                    else std::cout << v << " ";
                }
                std::cout << "\n";
            }
        } else {
            // печатаем несколько значений для проверки, чтобы показать
            std::cout << "Sample: d[0][1] = ";
            int v = global[idx(0, 1, N)];
            if (v >= INF / 2) std::cout << "INF\n";
            else std::cout << v << "\n";

            std::cout << "Sample: d[0][N-1] = ";
            v = global[idx(0, N - 1, N)];
            if (v >= INF / 2) std::cout << "INF\n";
            else std::cout << v << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
