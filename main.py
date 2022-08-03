from time import time

import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate._rbf import Rbf
from cs_rbf_with_faults import CsRbfWithFaults

from examples.example_1 import target, faults

# Сетка для рассчета значении интерполяционной функции в узлах сетки
x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)

x, y = np.meshgrid(x, y)

n, m = x.shape

# Истинное значение в узлах сетки, нужно только для тестирования (в продакшене исключить)
z = np.zeros(x.shape)

for i in range(0, n):
  for j in range(0, m):
    z[i, j] = target(x[i, j], y[i, j])

max_az = np.max(np.absolute(z))

fig = plt.figure()

# Кол-во точек где заданы значения для создания интерполяционной функции, в продакшане это len(points)
n = 1024

# Максимальная ошибка расчета
epsilon = 1e-4

# Радиус поиска соседних точек лежащих в окрестности поиска, нужно для CsRbfWithFaults
rs = 0.8 # 8 * math.sqrt(n_min / n)

# Точки где заданы значения для интерполяции, в продакшане нужно брать из БД
points = np.random.normal(0, 1, (n, 3))

for p in points:
  p[2] = target(p[0], p[1])

plt.title(f'RBF VS CS-RBF, n={n}')

ax = fig.add_subplot(1, 4, 1)
ax.set_xticks(np.arange(-1, 1.1, 0.4))
ax.set_yticks(np.arange(-1, 1.1, 0.2))
ax.scatter(points[:, 0], points[:, 1])
ax.grid()
ax.set_title('Scatter plot')

ax = fig.add_subplot(1, 4, 2, projection='3d')
ax.plot_surface(x, y, z,cmap='plasma')
ax.set_title('Surface plot')

# RBF интерполяции из библиотеки scipy, нужно только для сравнения с CsRbfWithFaults
start = time()
print('rbf start: ', start)

interpolator = Rbf(
  points[:, 0],
  points[:, 1],
  points[:, 2]  
)

z_rbf = interpolator(x, y)
mre = np.max(np.absolute(z-z_rbf)) / max_az

finish = time()
print('rbf finish: ', finish, ', time: ', finish - start)

ax = fig.add_subplot(1, 4, 3, projection='3d')
ax.plot_surface(x, y, z_rbf, cmap='plasma')
ax.set_title(f'Without Faults, mre={round(mre, 2)}, time={round(finish - start, 2)}')

# RBF интерполяция с учетом разломов
start = time()
print('rbf start: ', start)

# Интерполяция с учетом разломов
# points - точки где заданы значения для интерполяции, в продакшане нужно брать из БД
# faults - список линии разломов, формата [((x1, y1), (x2, y2)), ...], где
#   x1, y1 - координаты первой точки линии
#   x2, y2 - координаты второй точки линии
# epsilon - максимальная ошибка расчета
# rs - радиус поиска соседних точек лежащих в окрестности поиска, нужно для CsRbfWithFaults
interpolator = CsRbfWithFaults(
  points,
  faults,
  epsilon=epsilon,
  rs=rs  
)

z_cs_rbf = interpolator(x, y)
mre = np.max(np.absolute(z-z_cs_rbf)) / max_az

finish = time()
print('rbf finish: ', finish, ', time: ', finish - start)

ax = fig.add_subplot(1, 4, 4, projection='3d')
ax.plot_surface(x, y, z_cs_rbf, cmap='plasma')
ax.set_title(f'Without Faults, mre={round(mre, 2)}, time={round(finish - start, 2)}')

plt.show()
