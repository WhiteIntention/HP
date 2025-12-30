#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

// среднее (последовательно)
double average_seq(const int* a, int n) {
    long long sum = 0;
    for (int i = 0; i < n; i++) sum += a[i];
    return static_cast<double>(sum) / n;
}

// среднее (OpenMP reduction)
double average_omp(const int* a, int n) {
    long long sum = 0;

#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }

    return static_cast<double>(sum) / n;
}

int main() {
    const int N = 5'000'000; // размер массива

    // динамическая память через указатель
    int* arr = new int[N];

    // заполняем 1..100
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);

    for (int i = 0; i < N; i++) {
        arr[i] = dist(gen);
    }

    // последовательный подсчёт
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg1 = average_seq(arr, N);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_seq = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    // параллельный подсчёт
    auto t3 = std::chrono::high_resolution_clock::now();
    double avg2 = average_omp(arr, N);
    auto t4 = std::chrono::high_resolution_clock::now();
    auto ms_omp = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

    // вывод
    std::cout << "Part 3: Dynamic array average\n";
    std::cout << "N = " << N << "\n\n";

    std::cout << "Sequential:\n";
    std::cout << "Average = " << avg1 << "\n";
    std::cout << "Time (ms) = " << ms_seq << "\n\n";

    std::cout << "OpenMP reduction:\n";
    std::cout << "Average = " << avg2 << "\n";
    std::cout << "Time (ms) = " << ms_omp << "\n";
    std::cout << "Threads = " << omp_get_max_threads() << "\n";

    // освобождение памяти
    delete[] arr;

    return 0;
}
