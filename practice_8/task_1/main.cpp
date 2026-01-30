#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <omp.h>

// Обработка массива на CPU с использованием OpenMP
// Каждый элемент умножается на 2
void process_array_omp(std::vector<float>& data)
{
    int n = data.size();

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        data[i] *= 2.0f;
    }
}

int main()
{
    const std::vector<int> sizes = { 1'000'000, 10'000'000 };

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::cout << std::fixed << std::setprecision(6);

    for (int N : sizes) {
        std::vector<float> data(N);

        // Инициализация массива
        for (int i = 0; i < N; ++i) {
            data[i] = dist(rng);
        }

        auto start = std::chrono::high_resolution_clock::now();

        process_array_omp(data);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Минимальная проверка
        std::cout << "N = " << N << "\n";
        std::cout << "First element = " << data[0] << "\n";
        std::cout << "CPU OpenMP time (ms) = " << ms << "\n\n";
    }

    return 0;
}
