from typing import Any, Callable

import math

import numpy as np

from scipy.spatial import KDTree
from scipy.sparse.linalg import LinearOperator, cg

# нахождение расстояния между 2 точками на плоскости
# p1 - первая точка
# p2 - вторая точка
def distance(
  p1: tuple[float, float], 
  p2: tuple[float, float]
):
  return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# нахождение значение базисной функции по радиусу
# r - радиус, т.е. расстояние между двумя точками
# rs - радиус, больше которого базисная функция обнуляется
def basis(
  r: float, 
  rs: float
):
  return (4 * r / rs + 1) * (1 - r / rs) ** 4

# нахождение параметров прямой по 2 точкам на плоскости
# p1 - первая точка
# p2 - вторая точка
# error - максимальная ошибка расчета
def line(  
  p1: tuple[float, float],
  p2: tuple[float, float],
  error: float
):
  if abs(p2[0] - p1[0]) >= error:
    k = (p2[1] - p1[1])/(p2[0] - p1[0])
    b = (p1[1]*p2[0] - p2[1]*p1[0])/(p2[0] - p1[0])

    return (k, b)

  return None

# нахождение точки пересечения двух прямых лежащих на плоскости
# k1 - тангенс угла наклона первой прямой
# b1 - сдвиг по оси oy первой прямой
# k2 - тангенс угла наклона второй прямой
# b2 - сдвиг по оси oy второй прямой
# error - максимальная ошибка расчета
def lines_intersection(
  k1: float, 
  b1: float, 
  k2: float, 
  b2: float,
  error: float
):
  if abs(k2 - k1) >= error:
    x = (b1 - b2)/(k2 - k1)
    y = (k2*b1 - k1*b2)/(k2 - k1)

    return (x, y)

  return None

# Проверка, что точка p лежит на отрезке между двумя точками p1 и p2
# p - исследуемая точка
# p1 - первая точка
# p2 - вторая точка
def point_in_range(
  p: tuple[float, float], 
  p1: tuple[float, float], 
  p2: tuple[float, float]
):
  t = (p[0] - p1[0])/(p2[0] - p1[0])

  if t >= 0 and t <= 1:
    return True

  return False

# нахождение точки отстоящии на расстоянии r от точки p1 и лежащий на отрезке между точками p1 и p2
# p1 - первая точка
# p2 - вторая точка
# r - расстояние от точки p1
# l - расстояние между точками p1 и p2
def calculate_point(
  p1: tuple[float, float], 
  p2: tuple[float, float], 
  r: float, 
  l: float
):
  return (
    p1[0] + r * (p2[0] - p1[0]) / l,
    p1[1] + r * (p2[1] - p1[1]) / l
  )

# проверка, что точка p лежит выше прямой line_data
# p - исследуемая точка
# p1 - точка лежащая на прямой
# line_data - параметры прямой
## первый элемент - тангенс угла наклона прямой
## второй элемент - сдвиг по оси oy прямой
def point_up_line(
  p: tuple[float, float], 
  p1: tuple[float, float], 
  line_data: tuple[float, float] | None
):
  if line_data == None:
    if p[0] >= p1[0]:
      return True
    else:
      return False
  else:
    k, b = line_data

    y = k * p[0] + b

    if p[1] >= y:
      return True
    else:
      return False

# Проверка, что точки p1 и p2 лежат по разные стороны от линии разлома
# p1 - первая точка лежащая на линии line_data (линия разлома)
# p2 - вторая точка лежащая на линии line_data (линия разлома)
# pp1 - первая исследуемая точка лежащая на линии line_data_2
# pp2 - вторая исследуемая точка лежащая на линии line_data_2
# line_data - параметры первой прямой
## первый элемент - тангенс угла наклона прямой
## второй элемент - сдвиг по оси oy прямой
# line_data_2 - параметры второй прямой
## первый элемент - тангенс угла наклона прямой
## второй элемент - сдвиг по оси oy прямой
# error - максимальная ошибка расчета
def exclude_point(
  p1: tuple[float, float], 
  p2: tuple[float, float], 
  pp1: tuple[float, float], 
  pp2: tuple[float, float], 
  line_data: tuple[float, float] | None, 
  line_data_2: tuple[float, float] | None,
  error: float
):
  if line_data == None:
    if line_data_2 == None:
      if pp1[0] == p1[0]:
        y1 = p1[1]
        y2 = p2[1]

        if p1[1] > p2[1]:
          y1 = p2[1]
          y2 = p1[1]

        if (pp1[1] >= y1 and pp1[1] <= y2) or (pp2[1] >= y1 and pp2[1] <= y2):
          return True
    else:
      k2, b2 = line_data_2

      y = k2 * p1[0] + b2

      y1 = p1[1]
      y2 = p2[1]

      if p1[1] > p2[1]:
        y1 = p2[1]
        y2 = p1[1]

      if y >= y1 and y <= y2:
        return True
  else:
    k, b = line_data

    if line_data_2 == None:
      y = k * pp1[0] + b
      return point_in_range((pp1[0], y), p1, p2)
    else:
      k2, b2 = line_data_2

      p = lines_intersection(k, b, k2, b2, error)

      if p == None:
        return point_in_range(pp1, p1, p2) or point_in_range(pp2, p1, p2)
      else:
        return point_in_range(p, p1, p2)  

  return False

# исключаются все точки лежащие в окрестности радиуса rf с центром в точке pr линии разлома и находящие по разные стороны от линии разлома проходящее через точки p1 и p2
# pr - точка между p1 и p2 и отстоящий на расстоянии r от p1
# p1 - первая точка линии разлома
# p2 - вторая точка линии разлома
# tree - объект типа KDTree, нужен для быстрого (за время O(logN)) поиска точек лежащих в окрестности радиуса rf
# rf - радиус окрестности
# points - список точек
# res - список множества индексов точек лежащих в окрестности радиуса rf с центром в точках points, индекс списка совпадает с индексом списка points 
# error - максимальная ошибка расчета
def exclude_points(
  pr: tuple[float, float], 
  p1: tuple[float, float], 
  p2: tuple[float, float], 
  tree: KDTree, 
  rf: float, 
  points: list[tuple[float, float]], 
  res: list[set],
  error: float
):
  line_data = line(p1, p2, error)

  found = tree.query_ball_point(pr, rf)

  up = []
  down = []

  for i in found:
    p = points[i]

    if point_up_line(p, p1, line_data):
      up.append(i)
    else:
      down.append(i)

  for i in up:
    pp1 = points[i]

    for j in down:
      pp2 = points[j]

      line_data_2 = line(pp1, pp2, error)       

      if exclude_point(p1, p2, pp1, pp2, line_data, line_data_2, error):
        res[i].discard(j)
        res[j].discard(i) 

# исключаются все точки из res лежащие в окрестности радиуса rf с центром в точке pr линии разлома и находящие по разные стороны от линии разлома проходящее через точки p1 и p2 
# faults - линии разлома, список точек p1 и p2 линии разлома
# tree - объект типа KDTree, нужен для быстрого (за время O(logN)) поиска точек лежащих в окрестности радиуса rf
# rf - радиус окрестности
# points - список точек
# res - список множества индексов точек лежащих в окрестности радиуса rf с центром в точках points, индекс списка совпадает с индексом списка points
# ri - начальное расстояние от точки p1
# error - максимальная ошибка расчета
def faults_exclude_points(
  faults: list[tuple[tuple[float, float], tuple[float, float]]], 
  tree: KDTree, 
  rf: float, 
  points: list[tuple[float, float]], 
  res: list[set], 
  ri: float,
  error: float
):
  for p1, p2 in faults:
    l = distance(p1, p2)
      
    exclude_points(p1, p1, p2, tree, rf, points, res, error)

    r = ri

    while r < l:
      pr = calculate_point(p1, p2, r, l)

      exclude_points(pr, p1, p2, tree, rf, points, res, error)

      r += ri

    exclude_points(p2, p1, p2, tree, rf, points, res, error)

# исключаются все точки из found лежащие в окрестности радиуса rs с центром в исследуемой точке pp1 и находящие по разные стороны от линии разлома проходящее через точки p1 и p2 
# pp1 - исследуемая точка
# found - список точек лежащих в окрестности радиуса rs и с центром в точке pp1
# faults - линии разлома, список точек p1 и p2 линии разлома
# points - список точек
# error - максимальная ошибка расчета
def faults_exclude_point_from_found(
  pp1: tuple[float, float], 
  found: list, 
  points: list[tuple[float, float]], 
  faults: list[tuple[tuple[float, float], tuple[float, float]]],
  error: float
):
  res = set(found)

  for p1, p2 in faults:
    line_data = line(p1, p2, error)

    pp1_up = point_up_line(pp1, p1, line_data)

    for i in found:
      pp2 = points[i]

      pp2_up = point_up_line(pp2, p1, line_data)

      if pp1_up != pp2_up:
        line_data_2 = line(pp1, pp2, error)

        if exclude_point(p1, p2, pp1, pp2, line_data, line_data_2, error):
          res.discard(i)

  return res

# класс интерполятора с учетом линии разломов
class CsRbfWithFaults:
  # Расчет параметров для интерполятора с учетом разломов, используется для создания интерполятора, когда коэффициенты перед базисными функциями уже известны
  # rs - радиус поиска соседних точек лежащих в окрестности поиска, нужно для CsRbfWithFaults
  # epsilon - максимальная ошибка расчета
  # points - точки где заданы значения для интерполяции, в продакшане нужно брать из БД
  # faults - список линии разломов, формата [((x1, y1), (x2, y2)), ...], где
  #   x1, y1 - координаты первой точки линии
  #   x2, y2 - координаты второй точки линии
  # Результат:
  # tree - объект типа KDTree, нужен для быстрого (за время O(logN)) поиска точек лежащих в окрестности радиуса rs
  # res - список множества индексов точек лежащих в окрестности радиуса rs с центром в точках points, индекс списка совпадает с индексом списка points
  # db - словарь, где индекс из списка points ставит в соответствие индекс из списка fb, нужно для уменьшения времени расчета 
  # fb - список свободных членов для решения СЛАУ A*b = fb, нужно только для проверки точности расчитанных коэффициентов b, т.е. если b уже найдено, то для проверки его точности, нужно найти разность A*b - fb
  def check_faults(
    points: np.ndarray[Any, np.dtype[np.float64]],
    faults: list[tuple[tuple[float, float], tuple[float, float]]],
    epsilon: float = 1e-4,
    rs: float = 0.8
  ):
    n = len(points)

    rf = 2 * rs / math.sqrt(3)

    tree = KDTree(points[:, :2], copy_data=True)
    
    res = []

    for p in points:
      found = tree.query_ball_point(p[:2], rs)

      res.append(set(found))

    faults_exclude_points(faults, tree, rf, points, res, rs, rs * epsilon)

    fb = []
    db = {}

    for i in range(0, n):
      p = points[i]

      if len(res[i]) > 1:
        db[i] = len(fb)
        fb.append(p[2])

    return tree, res, db, fb    

  # Расчет коэффициентов перед базисными функциями и создание интерполятора
  # rs - радиус поиска соседних точек лежащих в окрестности поиска, нужно для CsRbfWithFaults
  # epsilon - максимальная ошибка расчета
  # points - точки где заданы значения для интерполяции, в продакшане нужно брать из БД
  # faults - список линии разломов, формата [((x1, y1), (x2, y2)), ...], где
  #   x1, y1 - координаты первой точки линии
  #   x2, y2 - координаты второй точки линии
  # rbf - пользовательская базисная функция
  def interpolator(
    points: np.ndarray[Any, np.dtype[np.float64]],
    faults: list[tuple[tuple[float, float], tuple[float, float]]],
    rbf: Callable[[float, float], float] = basis,
    epsilon: float = 1e-4,
    rs: float = 0.8
  ):
    n = len(points)

    tree, res, db, fb = CsRbfWithFaults.check_faults(points, faults, epsilon, rs)

    m = len(fb)

    f = np.zeros(m)

    def mv(v):      
      for k in range(m):
        f[k] = 0
      
      k = 0

      for i in range(n):
        if len(res[i]) > 1:
          for j in res[i]:
            a = rbf(
              distance(points[i], points[j]),
              rs
            )

            f[k] += a * v[db[j]]

          k += 1

      return f

    A = LinearOperator((m, m), matvec=mv)    

    b = cg(A, fb)

    return CsRbfWithFaults(
      rs, 
      epsilon,
      points,
      faults,
      b[0],
      db,
      res,
      tree,
      rbf
    )

  # Расчет значение интерполяционной функции в точке (x, y)
  # x - x координата точки
  # y - y координата точки
  def __call__(
    self,
    x: np.ndarray,
    y: np.ndarray
  ):
    n, m = x.shape

    z = np.zeros(x.shape)

    for i in range(0, n):
      for j in range(0, m):
        found = self.tree.query_ball_point((x[i, j], y[i, j]), self.rs)
        found = faults_exclude_point_from_found((x[i, j], y[i, j]), found, self.points, self.faults, self.rs * self.epsilon)  

        for k in found:
          l = self.db[k]
          p = self.points[k]
          z[i, j] += self.b[l] * self.rbf(distance((x[i, j], y[i, j]), p), self.rs)

    return z

  # Создание интерполятора в случае, когда коэффициенты b перед базисными функциями уже известны 
  # b - коэффициенты перед базисными функциями интерполяционной функции
  # rs - радиус поиска соседних точек лежащих в окрестности поиска, нужно для CsRbfWithFaults
  # epsilon - максимальная ошибка расчета
  # points - точки где заданы значения для интерполяции, в продакшане нужно брать из БД
  # faults - список линии разломов, формата [((x1, y1), (x2, y2)), ...], где
  #   x1, y1 - координаты первой точки линии
  #   x2, y2 - координаты второй точки линии
  # tree - объект типа KDTree, нужен для быстрого (за время O(logN)) поиска точек лежащих в окрестности радиуса rs
  # res - список множества индексов точек лежащих в окрестности радиуса rs с центром в точках points, индекс списка совпадает с индексом списка points
  # db - словарь, где индекс из списка points ставит в соответствие индекс из списка fb, нужно для уменьшения времени расчета
  def __init__(
    self,
    rs: float,
    epsilon: float,    
    points: np.ndarray[Any, np.dtype[np.float64]],    
    faults: list[tuple[tuple[float, float], tuple[float, float]]],
    b: np.ndarray[Any, np.float64],
    db: dict,
    res: list[set],
    tree: KDTree,
    rbf: Callable[[float, float], float] = basis
  ):
    self.rs = rs
    self.epsilon = epsilon

    self.rbf = rbf

    self.points = points
    self.faults = faults

    self.b = b
    self.db = db
    self.res = res
    self.tree = tree

    return 
