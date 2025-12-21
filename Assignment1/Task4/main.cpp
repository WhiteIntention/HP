#include <iostream>   
#include <random>     
#include <chrono>     
#include <iomanip>    
#include <omp.h>      

int main() {
    const std::size_t N = 5'000'000; // размер массива

    //создаём массив
    int* arr = new int[N];

    // заполняем случайными числами (1..10'000'000)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 10'000'000);

    for (std::size_t i = 0; i < N; ++i) {
        arr[i] = dist(gen);
    }

    // Последовательный подсчёт
    long long sum_seq = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < N; ++i) {
        sum_seq += arr[i];
    }
    double avg_seq = static_cast<double>(sum_seq) / static_cast<double>(N);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto ms_seq = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // Параллельный подсчёт (OpenMP reduction) 
    long long sum_par = 0;

    auto t3 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for reduction(+:sum_par)
    for (std::size_t i = 0; i < N; ++i) {
        sum_par += arr[i];
    }
    double avg_par = static_cast<double>(sum_par) / static_cast<double>(N);
    auto t4 = std::chrono::high_resolution_clock::now();

    auto ms_par = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    // 3) вывод результатов
    std::cout << "Task 4: Average (sequential vs OpenMP reduction)\n";
    std::cout << "N = " << N << "\n\n";

    std::cout << "Sequential:\n";
    std::cout << "  Sum = " << sum_seq << "\n";
    std::cout << "  Average = " << std::fixed << std::setprecision(4) << avg_seq << "\n";
    std::cout << "  Time (ms) = " << ms_seq << "\n\n";

    std::cout << "OpenMP reduction:\n";
    std::cout << "  Sum = " << sum_par << "\n";
    std::cout << "  Average = " << std::fixed << std::setprecision(4) << avg_par << "\n";
    std::cout << "  Time (ms) = " << ms_par << "\n";
    std::cout << "  Threads = " << omp_get_max_threads() << "\n\n";

    // 4) освобождение памяти
    delete[] arr;

    return 0;
}
