// lab3.cpp
// Динамический массив + среднее значение + параллелизация OpenMP
//
// 1) Создаёт динамический массив с помощью указателя (new[])
// 2) Заполняет случайными числами
// 3) Считает среднее последовательно и параллельно (OpenMP + reduction)
// 4) Освобождает память (delete[])

#include <iostream>
#include <random>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// Cреднеe значениe
double average_sequential(const int* arr, std::size_t n) {
    long long sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += arr[i];
    }
    return static_cast<double>(sum) / static_cast<double>(n);
}

// Параллельный расчёт среднего значения с OpenMP
double average_parallel(const int* arr, std::size_t n) {
    long long sum = 0;

#ifdef _OPENMP
    // Каждый поток считает свою локальную сумму, потом reduction складывает их
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < static_cast<int>(n); ++i) {
        sum += arr[i];
    }
#else
    // Если OpenMP не включён при компиляции, делаем обычный проход
    for (std::size_t i = 0; i < n; ++i) {
        sum += arr[i];
    }
#endif

    return static_cast<double>(sum) / static_cast<double>(n);
}

int main() {
    using namespace std;

    cout << "Enter N (array size): ";
    size_t N;
    if (!(cin >> N) || N == 0) {
        cerr << "Invalid size\n";
        return 1;
    }

    // 1. Динамический массив через указатель
    int* arr = new (nothrow) int[N];
    if (!arr) {
        cerr << "Memory allocation failed\n";
        return 1;
    }

    // 2. Заполнение случайными числами
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 100); // диапазон можно менять

    for (size_t i = 0; i < N; ++i) {
        arr[i] = dist(gen);
    }

    // Можно вывести первые несколько элементов для проверки
    const size_t PRINT_LIMIT = 20;
    cout << "First elements: ";
    for (size_t i = 0; i < min(N, PRINT_LIMIT); ++i) {
        cout << arr[i] << ' ';
    }
    if (N > PRINT_LIMIT) cout << "...";
    cout << "\n\n";

    // 3. Последовательное среднее
    auto t1 = chrono::high_resolution_clock::now();
    double avg_seq = average_sequential(arr, N);
    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> dur_seq = t2 - t1;

    // 3. Параллельное среднее
    auto t3 = chrono::high_resolution_clock::now();
    double avg_par = average_parallel(arr, N);
    auto t4 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> dur_par = t4 - t3;

    cout << "Sequential average: " << avg_seq
         << ", time = " << dur_seq.count() << " ms\n";
    cout << "Parallel   average: " << avg_par
         << ", time = " << dur_par.count() << " ms\n";

#ifdef _OPENMP
    cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";
#else
    cout << "OpenMP not enabled (compile with -fopenmp)\n";
#endif

    // 4. Освобождение памяти
    delete[] arr;
    arr = nullptr;

    return 0;
}
