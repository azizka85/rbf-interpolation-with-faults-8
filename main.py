from time import time

import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate._rbf import Rbf
from cs_rbf_with_faults import CsRbfWithFaults

# from examples.example_1 import target, faults
from examples.example_2 import target, faults

# Сетка для расчета значении интерполяционной функции в узлах сетки
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

# Устанавливается значения z для всех точек
for p in points:
  p[2] = target(p[0], p[1])

plt.title(f'RBF VS CS-RBF, n={n}')

# рисуются точки на плоскости
ax = fig.add_subplot(1, 5, 1)
ax.set_xticks(np.arange(-1, 1.1, 0.4))
ax.set_yticks(np.arange(-1, 1.1, 0.2))
ax.scatter(points[:, 0], points[:, 1])
ax.grid()
ax.set_title('Scatter plot')

# истинная поверхность заданная функцией target из examples
ax = fig.add_subplot(1, 5, 2, projection='3d')
ax.plot_surface(x, y, z,cmap='plasma')
ax.set_title('Surface plot')

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
mre = np.max(np.absolute(z-z_rbf)) / max_az

finish = time()
print('rbf finish: ', finish, ', time: ', finish - start)

# рисуется интерполяционная функция из библиотеки scipy
ax = fig.add_subplot(1, 5, 3, projection='3d')
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
interpolator = CsRbfWithFaults.interpolator(
  points,
  faults,
  epsilon=epsilon,
  rs=rs  
)

# нахождение значения z по точкам на плоскости через интерполятор учитывающий разломы
z_cs_rbf = interpolator(x, y)
mre = np.max(np.absolute(z-z_cs_rbf)) / max_az

finish = time()
print('rbf finish: ', finish, ', time: ', finish - start)

# рисуется интерполяционная функция через интерполятор учитывающий разломы
ax = fig.add_subplot(1, 5, 4, projection='3d')
ax.plot_surface(x, y, z_cs_rbf, cmap='plasma')
ax.set_title(f'Without Faults, mre={round(mre, 2)}, time={round(finish - start, 2)}')

# RBF интерполяция с учетом разломов
start = time()
print('rbf start: ', start)

# Интерполяция с учетом разломов без расчета коэффицентов перед базисными функциями, используется в случае когда эти коэффиценты уже известны
# rs - радиус поиска соседних точек лежащих в окрестности поиска, нужно для CsRbfWithFaults
# epsilon - максимальная ошибка расчета
# points - точки где заданы значения для интерполяции, в продакшане нужно брать из БД
# faults - список линии разломов, формата [((x1, y1), (x2, y2)), ...], где
#   x1, y1 - координаты первой точки линии
#   x2, y2 - координаты второй точки линии
# tree - объект типа KDTree, нужен для быстрого (за время O(logN)) поиска точек лежащих в окрестности радиуса rs
# res - список множества индексов точек лежащих в окрестности радиуса rs с центром в точках points, индекс списка совпадает с индексом списка points
# db - словарь, где индекс из списка points ставит в соответствие индекс из списка fb, нужно для уменьшения времени расчета 
# fb - список свободных членов для решения СЛАУ A*b = fb, нужно только для проверки точности расчитанных коэффициентов b, т.е. если b уже найдено, то для проверки его точности, нужно найти разность A*b - fb
tree, res, db, fb = CsRbfWithFaults.check_faults(
  points, 
  faults, 
  epsilon, 
  rs
)

# создание интерполятора, используется в случае когда коэффиценты перед базисными функциями уже расчитаны и известны
# b - коэффициенты интерполяционной функции
# rs - радиус поиска соседних точек лежащих в окрестности поиска, нужно для CsRbfWithFaults
# epsilon - максимальная ошибка расчета
# points - точки где заданы значения для интерполяции, в продакшане нужно брать из БД
# faults - список линии разломов, формата [((x1, y1), (x2, y2)), ...], где
#   x1, y1 - координаты первой точки линии
#   x2, y2 - координаты второй точки линии
# tree - объект типа KDTree, нужен для быстрого (за время O(logN)) поиска точек лежащих в окрестности радиуса rs
# res - список множества индексов точек лежащих в окрестности радиуса rs с центром в точках points, индекс списка совпадает с индексом списка points
# db - словарь, где индекс из списка points ставит в соответствие индекс из списка fb, нужно для уменьшения времени расчета 
interpolator = CsRbfWithFaults(
  rs,
  epsilon,  
  points,
  faults,
  interpolator.b,
  db,
  res,
  tree
)

# нахождение значения z по точкам на плоскости через интерполятор учитывающий разломы
z_cs_rbf = interpolator(x, y)
mre = np.max(np.absolute(z-z_cs_rbf)) / max_az

finish = time()
print('rbf finish: ', finish, ', time: ', finish - start)

# рисуется интерполяционная функция через интерполятор учитывающий разломы
ax = fig.add_subplot(1, 5, 5, projection='3d')
ax.plot_surface(x, y, z_cs_rbf, cmap='plasma')
ax.set_title(f'Without Faults, mre={round(mre, 2)}, time={round(finish - start, 2)}')

plt.show()
