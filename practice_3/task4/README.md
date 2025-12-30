# Сравнение производительности CPU и GPU (CUDA)

## Цель
Сравнить время выполнения сортировок на CPU и GPU (CUDA) для разных размеров массива.

Алгоритмы:
- Merge Sort (сортировка слиянием)
- Quick Sort (быстрая сортировка)
- Heap Sort (пирамидальная сортировка)

## Размеры данных
Тестирование проводится для:
- 10 000
- 100 000
- 1 000 000 элементов

## Как измеряется время
### CPU
Используется `std::chrono::high_resolution_clock`.

### GPU
Измеряются два времени:
- `kernel` - время выполнения CUDA-ядер (cudaEvent)
- `total` - общее время с копированием памяти (chrono)

## Файлы
- `benchmark.cu` - код бенчмарка (CPU + GPU)
- `block.png` - блок-схема
- `screenshot.png` - результаты запуска

## Компиляция (для NVIDIA T4):
```bash
nvcc -O2 benchmark.cu -o bench -gencode arch=compute_75,code=sm_75
