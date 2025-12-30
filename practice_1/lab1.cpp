#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

int main() {
    const int N = 10000; // размер массива (можно менять)

    int* a = new int[N];

    // заполняем 1..100
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);

    for (int i = 0; i < N; i++) a[i] = dist(gen);

    // вывод массива
    std::cout << "Array:\n";
    for (int i = 0; i < N; i++) {
        std::cout << a[i] << (i + 1 == N ? "\n" : " ");
    }
    std::cout << "\n";

    // ---------- последовательный min/max ----------
    int mn_seq = a[0];
    int mx_seq = a[0];

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 1; i < N; i++) {
        if (a[i] < mn_seq) mn_seq = a[i];
        if (a[i] > mx_seq) mx_seq = a[i];
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto us_seq = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // ---------- параллельный min/max (OpenMP) ----------
    int mn_par = a[0];
    int mx_par = a[0];

    auto t3 = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
        int local_min = a[0];
        int local_max = a[0];

#pragma omp for nowait
        for (int i = 0; i < N; i++) {
            if (a[i] < local_min) local_min = a[i];
            if (a[i] > local_max) local_max = a[i];
        }

#pragma omp critical
        {
            if (local_min < mn_par) mn_par = local_min;
            if (local_max > mx_par) mx_par = local_max;
        }
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    auto us_par = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    // результаты
    std::cout << "Sequential:\n";
    std::cout << "Min = " << mn_seq << "\n";
    std::cout << "Max = " << mx_seq << "\n";
    std::cout << "Time (us) = " << us_seq << "\n\n";

    std::cout << "OpenMP:\n";
    std::cout << "Min = " << mn_par << "\n";
    std::cout << "Max = " << mx_par << "\n";
    std::cout << "Time (us) = " << us_par << "\n";
    std::cout << "Threads = " << omp_get_max_threads() << "\n";

    delete[] a;
    return 0;
}
