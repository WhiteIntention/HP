# CUDA: Параллельная сортировка слиянием (chunks + pairwise merge)

## Описание
Реализована параллельная сортировка на CUDA:
1) массив делится на подмассивы (chunks), каждый chunk сортируется отдельным CUDA-блоком
2) далее выполняются проходы слияния по парам (pairwise merge), где длина run удваивается

## Как работает
- Kernel `sort_chunks`: каждый блок загружает свой chunk в shared memory и сортирует его
- Kernel `merge_pass`: сливает две отсортированные части длины run в одну (параллельно по потокам)
- После каждого прохода run *= 2, пока весь массив не станет отсортированным

## Файлы
- `main.cu` - код CUDA
- `block.png` - блок-схема (текст)
- `screenshot.png` - скрин вывода в консоли

## Сборка и запуск (Google Colab)
1) Включить GPU: Runtime → Change runtime type → GPU
2) Компиляция:
```bash
!nvcc -O2 main.cu -o app -gencode arch=compute_75,code=sm_75
