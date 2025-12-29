#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

void fill_array(int* a, int n, int seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(1, 100000);
    for (int i = 0; i < n; i++) a[i] = dist(gen);
}

void copy_array(const int* src, int* dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[i];
}

bool is_sorted(const int* a, int n) {
    for (int i = 1; i < n; i++) {
        if (a[i - 1] > a[i]) return false;
    }
    return true;
}

// последовательная сортировка выбором
void selection_sort_seq(int* a, int n) {
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[min_idx]) min_idx = j;
        }
        if (min_idx != i) std::swap(a[i], a[min_idx]);
    }
}

void selection_sort_omp(int* a, int n) {
    for (int i = 0; i < n - 1; i++) {
        int global_min_val = a[i];
        int global_min_idx = i;

#pragma omp parallel
        {
            int local_min_val = a[i];
            int local_min_idx = i;

#pragma omp for nowait
            for (int j = i + 1; j < n; j++) {
                if (a[j] < local_min_val) {
                    local_min_val = a[j];
                    local_min_idx = j;
                }
            }

#pragma omp critical
            {
                if (local_min_val < global_min_val) {
                    global_min_val = local_min_val;
                    global_min_idx = local_min_idx;
                }
            }
        }

        if (global_min_idx != i) std::swap(a[i], a[global_min_idx]);
    }
}

long long time_us_seq(int* a, int n) {
    auto t1 = std::chrono::high_resolution_clock::now();
    selection_sort_seq(a, n);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

long long time_us_omp(int* a, int n) {
    auto t1 = std::chrono::high_resolution_clock::now();
    selection_sort_omp(a, n);
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

void run_test(int n) {
    int* base = new int[n];
    int* a1 = new int[n];
    int* a2 = new int[n];

    fill_array(base, n, 123);
    copy_array(base, a1, n);
    copy_array(base, a2, n);

    long long t_seq = time_us_seq(a1, n);
    long long t_omp = time_us_omp(a2, n);

    std::cout << "N = " << n << "\n";
    std::cout << "Seq time (us) = " << t_seq << "\n";
    std::cout << "OMP time (us) = " << t_omp << "\n";
    std::cout << "Seq sorted = " << (is_sorted(a1, n) ? "YES" : "NO") << "\n";
    std::cout << "OMP sorted = " << (is_sorted(a2, n) ? "YES" : "NO") << "\n";
    std::cout << "Threads = " << omp_get_max_threads() << "\n\n";

    delete[] base;
    delete[] a1;
    delete[] a2;
}

int main() {
    std::cout << "Task 3: Selection sort (sequential vs OpenMP)\n\n";

    run_test(1000);
    run_test(10000);

    return 0;
}
