%%writefile 9_2.cpp
#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>

// Индексация для локальной матрицы (rows_per_proc x N)
static inline double& Aat(std::vector<double>& A, int local_row, int col, int N) {
    return A[(size_t)local_row * (size_t)N + (size_t)col];
}
static inline const double& Aat(const std::vector<double>& A, int local_row, int col, int N) {
    return A[(size_t)local_row * (size_t)N + (size_t)col];
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Размер матрицы задаём параметром программы
    // Пример: mpirun -np 4 ./9_2 256
    int N = 128;
    if (argc >= 2) {
        N = std::max(1, std::atoi(argv[1]));
    }

    // Scatter требует одинаковое число строк на процесс -> делаем padding
    int rows_per_proc = (N + size - 1) / size;     // ceil(N/size)
    int Npad = rows_per_proc * size;               // сколько строк реально раздадим

    // На rank 0 создаём A и b. Чтобы было проще проверять, строим систему с известным x_true.
    std::vector<double> Aglobal;
    std::vector<double> bglobal;
    std::vector<double> x_true;

    if (rank == 0) {
        Aglobal.assign((size_t)Npad * (size_t)N, 0.0);
        bglobal.assign((size_t)Npad, 0.0);
        x_true.assign((size_t)N, 0.0);

        std::mt19937_64 rng(123);
        std::uniform_real_distribution<double> distA(-1.0, 1.0);
        std::uniform_real_distribution<double> distX(0.0, 1.0);

        for (int i = 0; i < N; ++i) x_true[(size_t)i] = distX(rng);

        // Генерим диагонально-доминантную матрицу, чтобы не мучаться с нулевыми pivot
        for (int i = 0; i < N; ++i) {
            double rowsum = 0.0;
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                double v = distA(rng);
                Aglobal[(size_t)i * (size_t)N + (size_t)j] = v;
                rowsum += std::fabs(v);
            }
            Aglobal[(size_t)i * (size_t)N + (size_t)i] = rowsum + 1.0; // доминантная диагональ
        }

        // b = A * x_true
        for (int i = 0; i < N; ++i) {
            double s = 0.0;
            for (int j = 0; j < N; ++j) {
                s += Aglobal[(size_t)i * (size_t)N + (size_t)j] * x_true[(size_t)j];
            }
            bglobal[(size_t)i] = s;
        }

        // padded строки уже нули (A=0, b=0) — их просто игнорируем при k < N
    }

    // Локальные куски: rows_per_proc строк матрицы и b
    std::vector<double> Alocal((size_t)rows_per_proc * (size_t)N, 0.0);
    std::vector<double> blocal((size_t)rows_per_proc, 0.0);

    // Таймер (будем мерить основную часть: Scatter + прямой ход + Gather)
    double start_time = 0.0;
    double end_time = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Scatter строк матрицы (каждый процесс получает rows_per_proc * N элементов)
    MPI_Scatter(
        rank == 0 ? Aglobal.data() : nullptr,
        rows_per_proc * N, MPI_DOUBLE,
        Alocal.data(),
        rows_per_proc * N, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Scatter вектора b (rows_per_proc элементов)
    MPI_Scatter(
        rank == 0 ? bglobal.data() : nullptr,
        rows_per_proc, MPI_DOUBLE,
        blocal.data(),
        rows_per_proc, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Буфер для pivot строки, которую будем рассылать
    std::vector<double> pivot_row((size_t)N, 0.0);
    double pivot_b = 0.0;

    // Прямой ход Гаусса
    for (int k = 0; k < N; ++k) {
        int owner = k / rows_per_proc;
        int owner_local = k % rows_per_proc;

        // Владелец pivot-строки кладёт её в буфер
        if (rank == owner) {
            for (int j = 0; j < N; ++j) {
                pivot_row[(size_t)j] = Aat(Alocal, owner_local, j, N);
            }
            pivot_b = blocal[(size_t)owner_local];
        }

        // Рассылаем pivot строку всем процессам
        MPI_Bcast(pivot_row.data(), N, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        double piv = pivot_row[(size_t)k];

        // Мы специально делали диагонально-доминантную матрицу, но на всякий случай:
        if (std::fabs(piv) < 1e-15) {
            if (rank == 0) {
                std::cerr << "Zero/near-zero pivot at k=" << k << ". Try another seed or add pivoting.\n";
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // У каждого процесса обновляем только свои строки с глобальным индексом i > k
        for (int lr = 0; lr < rows_per_proc; ++lr) {
            int gi = rank * rows_per_proc + lr;  // глобальный индекс строки
            if (gi <= k || gi >= N) continue;    // не трогаем pivot и padded строки

            double factor = Aat(Alocal, lr, k, N) / piv;

            // A[i, j] -= factor * pivot_row[j]
            Aat(Alocal, lr, k, N) = 0.0; // по идее должно стать нулём
            for (int j = k + 1; j < N; ++j) {
                Aat(Alocal, lr, j, N) -= factor * pivot_row[(size_t)j];
            }

            // b[i] -= factor * pivot_b
            blocal[(size_t)lr] -= factor * pivot_b;
        }
    }

    // Собираем обратно на rank 0 (матрица и b после прямого хода)
    MPI_Gather(
        Alocal.data(),
        rows_per_proc * N, MPI_DOUBLE,
        rank == 0 ? Aglobal.data() : nullptr,
        rows_per_proc * N, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    MPI_Gather(
        blocal.data(),
        rows_per_proc, MPI_DOUBLE,
        rank == 0 ? bglobal.data() : nullptr,
        rows_per_proc, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );
    //конец таймера
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        // Обратный ход на rank 0
        std::vector<double> x((size_t)N, 0.0);

        for (int i = N - 1; i >= 0; --i) {
            double diag = Aglobal[(size_t)i * (size_t)N + (size_t)i];
            double s = bglobal[(size_t)i];

            for (int j = i + 1; j < N; ++j) {
                s -= Aglobal[(size_t)i * (size_t)N + (size_t)j] * x[(size_t)j];
            }

            if (std::fabs(diag) < 1e-15) {
                std::cerr << "Zero/near-zero diagonal at i=" << i << "\n";
                MPI_Abort(MPI_COMM_WORLD, 2);
            }

            x[(size_t)i] = s / diag;
        }

        // Небольшая проверка, если мы строили систему из x_true
        double max_abs_err = 0.0;
        for (int i = 0; i < N; ++i) {
            max_abs_err = std::max(max_abs_err, std::fabs(x[(size_t)i] - x_true[(size_t)i]));
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "N = " << N << "\n";
        std::cout << "MPI processes = " << size << "\n";
        std::cout << "Max abs error vs x_true = " << max_abs_err << "\n";
        std::cout << "Execution time: " << (end_time - start_time) << " seconds\n";
        // Можно вывести первые несколько значений решения
        int show = std::min(N, 5);
        std::cout << "x[0.." << (show - 1) << "] = ";
        for (int i = 0; i < show; ++i) {
            std::cout << x[(size_t)i] << (i + 1 == show ? "" : ", ");
        }
        std::cout << "\n";
    }

    MPI_Finalize();
    return 0;
}
