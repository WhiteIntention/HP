#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

int main() {
    const int N = 10000; // размер массива

    // создание массива
    int* arr = new int[N];

    // генерация случайных чисел
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100000);

    for (int i = 0; i < N; i++) {
        arr[i] = dist(gen);
    }

    // ---------- последовательная версия ----------
    int min_seq = arr[0];
    int max_seq = arr[0];

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 1; i < N; i++) {
        if (arr[i] < min_seq) min_seq = arr[i];
        if (arr[i] > max_seq) max_seq = arr[i];
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto time_seq = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // ---------- параллельная версия ----------
    int min_par = arr[0];
    int max_par = arr[0];

    auto t3 = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
        int local_min = arr[0];
        int local_max = arr[0];

#pragma omp for nowait
        for (int i = 0; i < N; i++) {
            if (arr[i] < local_min) local_min = arr[i];
            if (arr[i] > local_max) local_max = arr[i];
        }

#pragma omp critical
        {
            if (local_min < min_par) min_par = local_min;
            if (local_max > max_par) max_par = local_max;
        }
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    auto time_par = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    // вывод результатов
    std::cout << "Task 2: Min/Max with OpenMP\n\n";

    std::cout << "Sequential version:\n";
    std::cout << "Min = " << min_seq << "\n";
    std::cout << "Max = " << max_seq << "\n";
    std::cout << "Time (us) = " << time_seq << "\n\n";

    std::cout << "Parallel OpenMP version:\n";
    std::cout << "Min = " << min_par << "\n";
    std::cout << "Max = " << max_par << "\n";
    std::cout << "Time (us) = " << time_par << "\n";
    std::cout << "Threads = " << omp_get_max_threads() << "\n";

    // освобождение памяти
    delete[] arr;

    return 0;
}
