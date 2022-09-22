from time import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate._rbf import Rbf
from cs_rbf_with_faults import distance, CsRbfWithFaults

from examples.example_3 import load_data

# загрузка данных
min_x, min_y, max_x, max_y, points, faults = load_data()

points = np.array(points)

# Сетка для расчета значении интерполяционной функции в узлах сетки
x = np.linspace(min_x, max_x, 101)
y = np.linspace(min_y, max_y, 101)

x, y = np.meshgrid(x, y)

n, m = x.shape

# Максимальная ошибка расчета
epsilon = 1e-4

# Радиус поиска соседних точек лежащих в окрестности поиска, нужно для CsRbfWithFaults
rs = 0.4 * distance(
  (min_x, min_y),
  (max_x, max_y)
)

k = len(points)

fig = plt.figure()

plt.title(f'RBF VS CS-RBF, n={k}')

# рисуются точки на плоскости
ax = fig.add_subplot(1, 3, 1)
ax.set_xticks(np.arange(min_x, max_x, (max_x - min_x) / 10))
ax.set_yticks(np.arange(min_y, max_y, (max_y - min_y) / 10))
ax.scatter(points[:, 0], points[:, 1])
ax.grid()
ax.set_title('Scatter plot')

# RBF интерполяции из библиотеки scipy, нужно только для сравнения с CsRbfWithFaults
start = time()
print('rbf start: ', start)

# инициализация интерполятора из библиотеки scipy
interpolator = Rbf(
  points[:, 0],
  points[:, 1],
  points[:, 2]  
)

# нахождение значения z по точкам на плоскости через интерполятор из библиотеки scipy
z_rbf = interpolator(x, y)

finish = time()
print('rbf finish: ', finish, ', time: ', finish - start)

# рисуется интерполяционная функция из библиотеки scipy
ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.plot_surface(x, y, z_rbf, cmap='plasma')
ax.set_title(f'Without Faults, time={round(finish - start, 2)}')

# RBF интерполяция с учетом разломов
start = time()
print('rbf start: ', start)

print(faults)

# Интерполяция с учетом разломов
# points - точки где заданы значения для интерполяции, в продакшане нужно брать из БД
# faults - список линии разломов, формата [((x1, y1), (x2, y2)), ...], где
#   x1, y1 - координаты первой точки линии
#   x2, y2 - координаты второй точки линии
# epsilon - максимальная ошибка расчета
# rs - радиус поиска соседних точек лежащих в окрестности поиска, нужно для CsRbfWithFaults
interpolator = CsRbfWithFaults.interpolator(
  points,
  faults,
  epsilon=epsilon,
  rs=rs  
)

# нахождение значения z по точкам на плоскости через интерполятор учитывающий разломы
z_cs_rbf = interpolator(x, y)

finish = time()
print('rbf finish: ', finish, ', time: ', finish - start)

# рисуется интерполяционная функция через интерполятор учитывающий разломы
ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.plot_surface(x, y, z_cs_rbf, cmap='plasma')
ax.set_title(f'Without Faults, time={round(finish - start, 2)}')

plt.show()
