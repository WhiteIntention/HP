#include <omp.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>

struct Stats {
    double sum = 0.0;
    double mean = 0.0;
    double var  = 0.0; // дисперсия (population)
};

// sum, mean, variance
Stats stats_sequential(const std::vector<double>& a, double& t_compute) {
    double t0 = omp_get_wtime();

    double sum = 0.0;
    double sumsq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum   += a[i];
        sumsq += a[i] * a[i];
    }

    double mean = sum / (double)a.size();
    double var  = sumsq / (double)a.size() - mean * mean;

    double t1 = omp_get_wtime();
    t_compute = (t1 - t0);

    return {sum, mean, var};
}

// Параллельно: reduction по sum и sumsq
Stats stats_parallel(const std::vector<double>& a, int threads, double& t_compute) {
    double t0 = omp_get_wtime();

    double sum = 0.0;
    double sumsq = 0.0;

    #pragma omp parallel for num_threads(threads) reduction(+:sum,sumsq) schedule(static)
    for (long long i = 0; i < (long long)a.size(); ++i) {
        double x = a[(size_t)i];
        sum   += x;
        sumsq += x * x;
    }

    double mean = sum / (double)a.size();
    double var  = sumsq / (double)a.size() - mean * mean;

    double t1 = omp_get_wtime();
    t_compute = (t1 - t0);

    return {sum, mean, var};
}

// Оценка последовательной доли по закону Амдала:
// S = 1 / ( s + (1-s)/p )  =>  s = (1/S - 1/p) / (1 - 1/p)
double amdahl_serial_fraction(double speedup, int p) {
    if (p <= 1) return 1.0;
    double invS = 1.0 / speedup;
    double invP = 1.0 / (double)p;
    return (invS - invP) / (1.0 - invP);
}

int main(int argc, char** argv) {
    // 10 млн элементов
    const size_t N = (argc >= 2) ? (size_t)std::max(1LL, std::stoll(argv[1])) : 10'000'000;
    const int max_threads = (argc >= 3) ? std::max(1, std::stoi(argv[2])) : std::min(16, omp_get_max_threads());

    std::cout << "Practical Work #10 (OpenMP): sum/mean/variance\n";
    std::cout << "N = " << N << ", max_threads = " << max_threads << "\n\n";

    // 1) Подготовка данных + измерение
    double t_init0 = omp_get_wtime();

    std::vector<double> a(N);
    std::mt19937_64 gen(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < N; ++i) a[i] = dist(gen);

    double t_init1 = omp_get_wtime();
    double t_init = (t_init1 - t_init0);

    // 2) Последовательная версия (baseline)
    double t_seq = 0.0;
    Stats s_seq = stats_sequential(a, t_seq);

    // 3) Параллельная версия: прогон по потокам
    // Берём набор 1,2,4,8,... плюс max_threads (если не степень двойки)
    std::vector<int> thread_list;
    for (int t = 1; t <= max_threads; t *= 2) thread_list.push_back(t);
    if (thread_list.back() != max_threads) thread_list.push_back(max_threads);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Init time (s)          = " << t_init << "\n";
    std::cout << "Sequential compute (s) = " << t_seq << "\n";
    std::cout << "Seq result: sum=" << s_seq.sum << " mean=" << s_seq.mean << " var=" << s_seq.var << "\n\n";

    std::cout << "Threads | Par time (s) | Speedup | Eff(%) | Amdahl serial s | Parallel part (1-s)\n";

    for (int t : thread_list) {
        double t_par = 0.0;
        Stats s_par = stats_parallel(a, t, t_par);

        // Проверка корректности
        double max_err = std::max({std::fabs(s_par.sum - s_seq.sum),
                                   std::fabs(s_par.mean - s_seq.mean),
                                   std::fabs(s_par.var - s_seq.var)});
        bool ok = max_err < 1e-6;

        double speedup = t_seq / t_par;
        double eff = (speedup / (double)t) * 100.0;

        double s = amdahl_serial_fraction(speedup, t);
        s = std::clamp(s, 0.0, 1.0);

        std::cout << std::setw(7) << t << " | "
                  << std::setw(10) << t_par << " | "
                  << std::setw(7) << speedup << " | "
                  << std::setw(6) << eff << " | "
                  << std::setw(15) << s << " | "
                  << std::setw(18) << (1.0 - s)
                  << (ok ? "" : "  (WARN: mismatch)") << "\n";
    }
    return 0;
}
