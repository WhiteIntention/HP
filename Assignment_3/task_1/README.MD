# Assignment 3 - Task 1 (CUDA): Element-wise multiply (Global vs Shared)

В этом задании реализована программа на **CUDA C++** для поэлементной обработки массива: умножение каждого элемента на константу `k`. Реализованы **две версии**:
1) **Только глобальная память (global memory)** - каждый поток читает `in[idx]` из global и записывает `out[idx]` в global.  
2) **С использованием разделяемой памяти (shared memory)** - каждый блок загружает элементы из global в shared, выполняет умножение в shared и записывает результат обратно в global.

Цель: сравнить время выполнения обеих реализаций для массива размера **N = 1 000 000**.

## Реализация

### Версия 1 - Global memory
Каждый поток выполняет:
- чтение `in[idx]` из global memory,
- умножение на `k`,
- запись `out[idx]` в global memory.

Kernel:
- `mul_global(const float* in, float* out, float k, int n)`

### Версия 2 - Shared memory
Внутри каждого блока:
- загрузка `in[idx]` в `shared`,
- умножение элемента `shared[tid] *= k`,
- запись обратно в `out[idx]`.

Kernel:
- `mul_shared(const float* in, float* out, float k, int n)`
- используется динамическая shared memory: `extern __shared__ float s[];`

## Параметры эксперимента

- Размер массива: `N = 1 000 000`
- Константа: `k = 3.5`
- Размер блока: `block = 256`
- Размер сетки: `grid = 3907`
- Shared memory per block: `256 * sizeof(float)` = **1024 bytes**
- Тайминг: среднее время kernel за **200 запусков** (cudaEvent)

Корректность проверялась сравнением с CPU-референсом:
- `Max abs error` (макс. абсолютная ошибка)
- допуск `EPS = 1e-4`

## Результаты (полученные измерения)
Вывод программы:
Assignment 3 - Task 1 (CUDA): element-wise multiply
N = 1000000, k = 3.5
Block = 256, Grid = 3907
Global memory kernel avg time (ms) = 0.037922
Shared memory kernel avg time (ms) = 0.045088
Max abs error = 0.000000
Correct = YES
Speed ratio (shared/global) = 1.188988

### Интерпретация
- **Global memory версия быстрее**: 0.037922 ms
- **Shared memory версия медленнее**: 0.045088 ms
- Замедление shared относительно global: **~1.19×** (`shared/global = 1.188988`)
- Корректность подтверждена: `Correct = YES`, `Max abs error = 0.0`

## Вывод

Для операции *поэлементного умножения* использование shared memory **не даёт ускорения** и приводит к ухудшению производительности. Причина в том, что каждый элемент входного массива используется **только один раз** (нет переиспользования данных между потоками), поэтому shared memory не помогает скрыть задержки чтения, а наоборот добавляет накладные расходы:
- дополнительные копирования `global → shared → global`,
- синхронизации `__syncthreads()`.

Shared memory эффективно применяется в задачах, где данные многократно переиспользуются внутри блока (например, **умножение матриц с tiling**, свёртки и т.п.), но для простой element-wise операции глобальная память оказывается оптимальнее.

## Сборка и запуск (Google Colab)
Компиляция (пример для Tesla T4):
```bash
!nvcc -O3 -std=c++17 -arch=sm_75 ass3_task1.cu -o ass3_task1
