#include <iostream>
#include <random>

// пузырьковая сортировка
void bubble_sort(int* a, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                std::swap(a[j], a[j + 1]);
            }
        }
    }
}

// сортировка выбором
void selection_sort(int* a, int n) {
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[min_idx]) min_idx = j;
        }
        std::swap(a[i], a[min_idx]);
    }
}

// сортировка вставками
void insertion_sort(int* a, int n) {
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

// печать массива
void print_array(const int* a, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    const int N = 20; // размер массива

    int* arr = new int[N];

    // заполнение случайными числами
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);

    for (int i = 0; i < N; i++) {
        arr[i] = dist(gen);
    }

    std::cout << "Original array:\n";
    print_array(arr, N);

    // пузырёк
    int* a1 = new int[N];
    std::copy(arr, arr + N, a1);
    bubble_sort(a1, N);
    std::cout << "\nBubble sort:\n";
    print_array(a1, N);

    // выбор
    int* a2 = new int[N];
    std::copy(arr, arr + N, a2);
    selection_sort(a2, N);
    std::cout << "\nSelection sort:\n";
    print_array(a2, N);

    // вставки
    int* a3 = new int[N];
    std::copy(arr, arr + N, a3);
    insertion_sort(a3, N);
    std::cout << "\nInsertion sort:\n";
    print_array(a3, N);

    delete[] arr;
    delete[] a1;
    delete[] a2;
    delete[] a3;

    return 0;
}
