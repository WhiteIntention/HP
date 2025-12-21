#include <iostream>   
#include <random>     
#include <chrono>     
#include <omp.h>      

int main() {
    const std::size_t N = 1'000'000; // размер массива

    // динамическое выделение памяти
    int* arr = new int[N];

    // генератор случайных чисел
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 1'000'000);

    // заполнение массива
    for (std::size_t i = 0; i < N; ++i) {
        arr[i] = dist(gen);
    }

    int global_min = arr[0];
    int global_max = arr[0];

    // начало измерения времени
    auto t1 = std::chrono::high_resolution_clock::now();

    // параллельный поиск минимума и максимума
#pragma omp parallel
    {
        int local_min = arr[0];
        int local_max = arr[0];

#pragma omp for nowait
        for (std::size_t i = 0; i < N; ++i) {
            if (arr[i] < local_min) local_min = arr[i];
            if (arr[i] > local_max) local_max = arr[i];
        }

#pragma omp critical
        {
            if (local_min < global_min) global_min = local_min;
            if (local_max > global_max) global_max = local_max;
        }
    }

    // конец измерения времени
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    // вывод результатов
    std::cout << "Task 3: Parallel min/max (OpenMP)\n";
    std::cout << "N = " << N << "\n";
    std::cout << "Min = " << global_min << "\n";
    std::cout << "Max = " << global_max << "\n";
    std::cout << "Time (ms) = " << ms << "\n";
    std::cout << "Threads = " << omp_get_max_threads() << "\n";

    // освобождение памяти
    delete[] arr;

    return 0;
}
