#include <iostream>   
#include <random>     
#include <chrono>     

int main() {
    const std::size_t N = 1'000'000; // размер массива

    // динамическое выделение памяти
    int* arr = new int[N];

    // настройка генератора случайных чисел
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 1'000'000);

    // заполнение массива случайными числами
    for (std::size_t i = 0; i < N; ++i) {
        arr[i] = dist(gen);
    }

    // начало измерения времени
    auto t1 = std::chrono::high_resolution_clock::now();

    // начальные значения минимума и максимума
    int mn = arr[0];
    int mx = arr[0];

    // последовательный поиск min и max
    for (std::size_t i = 1; i < N; ++i) {
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
    }

    // конец измерения времени
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    // вывод результатов
    std::cout << "Task 2: Sequential min/max\n";
    std::cout << "N = " << N << "\n";
    std::cout << "Min = " << mn << "\n";
    std::cout << "Max = " << mx << "\n";
    std::cout << "Time (ms) = " << ms << "\n";

    // освобождение памяти
    delete[] arr;

    return 0;
}
