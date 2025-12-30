#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <omp.h>

using clk = std::chrono::high_resolution_clock;

// генерация
std::vector<int> make_data(int n, int seed = 123) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(1, 1000000);
    std::vector<int> a(n);
    for (int i = 0; i < n; i++) a[i] = dist(gen);
    return a;
}

bool is_sorted_ok(const std::vector<int>& a) {
    for (int i = 1; i < (int)a.size(); i++)
        if (a[i - 1] > a[i]) return false;
    return true;
}

// последовательные версии
void bubble_seq(std::vector<int>& a) {
    int n = (int)a.size();
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - 1 - i; j++)
            if (a[j] > a[j + 1]) std::swap(a[j], a[j + 1]);
}

void selection_seq(std::vector<int>& a) {
    int n = (int)a.size();
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[min_idx]) min_idx = j;
        std::swap(a[i], a[min_idx]);
    }
}

void insertion_seq(std::vector<int>& a) {
    int n = (int)a.size();
    for (int i = 1; i < n; i++) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

// 1) Пузырёк
void bubble_omp_oddeven(std::vector<int>& a) {
    int n = (int)a.size();
    for (int phase = 0; phase < n; phase++) {
        int start = (phase % 2 == 0) ? 0 : 1;

#pragma omp parallel for
        for (int i = start; i < n - 1; i += 2) {
            if (a[i] > a[i + 1]) std::swap(a[i], a[i + 1]);
        }
    }
}

// 2) параллелим поиск минимума внутри, а i идёт последовательно.
void selection_omp(std::vector<int>& a) {
    int n = (int)a.size();
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        int min_val = a[i];

#pragma omp parallel
        {
            int local_idx = min_idx;
            int local_val = min_val;

#pragma omp for nowait
            for (int j = i + 1; j < n; j++) {
                if (a[j] < local_val) {
                    local_val = a[j];
                    local_idx = j;
                }
            }

#pragma omp critical
            {
                if (local_val < min_val) {
                    min_val = local_val;
                    min_idx = local_idx;
                }
            }
        }

        std::swap(a[i], a[min_idx]);
    }
}

// Чтобы показать OpenMP - делаем “параллельную” версию
// сортируем куски вставками
void insertion_sort_chunk(std::vector<int>& a, int l, int r) {
    for (int i = l + 1; i < r; i++) {
        int key = a[i];
        int j = i - 1;
        while (j >= l && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

void merge_two(const std::vector<int>& src, std::vector<int>& dst, int l, int m, int r) {
    int i = l, j = m, k = l;
    while (i < m && j < r) {
        if (src[i] <= src[j]) dst[k++] = src[i++];
        else dst[k++] = src[j++];
    }
    while (i < m) dst[k++] = src[i++];
    while (j < r) dst[k++] = src[j++];
}

void insertion_omp_chunks(std::vector<int>& a) {
    int n = (int)a.size();
    int threads = omp_get_max_threads();
    int chunks = threads; // по числу потоков
    int step = (n + chunks - 1) / chunks;

#pragma omp parallel for
    for (int c = 0; c < chunks; c++) {
        int l = c * step;
        int r = std::min(n, l + step);
        if (l < r) insertion_sort_chunk(a, l, r);
    }

    std::vector<int> tmp(n);
    for (int size = step; size < n; size *= 2) {
        for (int l = 0; l < n; l += 2 * size) {
            int m = std::min(n, l + size);
            int r = std::min(n, l + 2 * size);
            merge_two(a, tmp, l, m, r);
        }
        a.swap(tmp);
    }
}

template <class F>
long long time_ms(F func, std::vector<int> a) {
    auto t1 = clk::now();
    func(a);
    auto t2 = clk::now();
    if (!is_sorted_ok(a)) return -1;
    return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
}

int main() {
    std::vector<int> sizes = {1000, 10000, 100000};

    std::cout << "Practice 2: Sorting with OpenMP\n";
    std::cout << "Threads = " << omp_get_max_threads() << "\n\n";

    for (int n : sizes) {
        auto base = make_data(n, 123);

        std::cout << "N = " << n << "\n";

        // bubble
        auto b_seq = time_ms(bubble_seq, base);
        auto b_omp = time_ms(bubble_omp_oddeven, base);
        std::cout << "Bubble:   seq(ms)=" << b_seq << "  omp(ms)=" << b_omp << "\n";

        // selection
        auto s_seq = time_ms(selection_seq, base);
        auto s_omp = time_ms(selection_omp, base);
        std::cout << "Select:   seq(ms)=" << s_seq << "  omp(ms)=" << s_omp << "\n";

        // insertion
        auto i_seq = time_ms(insertion_seq, base);
        auto i_omp = time_ms(insertion_omp_chunks, base);
        std::cout << "Insert:   seq(ms)=" << i_seq << "  omp(ms)=" << i_omp << "\n\n";
    }

    return 0;
}

