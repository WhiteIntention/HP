#include <iostream>   
#include <random>     
#include <iomanip>    
#include <new>        

int main() {
    const std::size_t N = 50000;   // размер массива

    // динамическое выделение памяти
    int* arr = new int[N];

    // генератор случайных чисел от 1 до 100
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);

    long long sum = 0; // сумма элементов массива

    // заполнение массива и подсчёт суммы
    for (std::size_t i = 0; i < N; ++i) {
        arr[i] = dist(gen);
        sum += arr[i];
    }

    // вычисление среднего значения
    double avg = static_cast<double>(sum) / static_cast<double>(N);

    // вывод результатов
    std::cout << "Task 1: Dynamic array average\n";
    std::cout << "N = " << N << "\n";
    std::cout << "Sum = " << sum << "\n";
    std::cout << "Average = " << std::fixed << std::setprecision(3) << avg << "\n";

    // освобождение памяти
    delete[] arr;

    return 0;
}
