from typing import Any, Callable

import math

import numpy as np

from scipy.spatial import KDTree
from scipy.sparse.linalg import LinearOperator, cg

def distance(
  p1: tuple[float, float], 
  p2: tuple[float, float]
):
  return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def basis(
  r: float, 
  rs: float
):
  return (4 * r / rs + 1) * (1 - r / rs) ** 4

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

def point_in_range(
  p: tuple[float, float], 
  p1: tuple[float, float], 
  p2: tuple[float, float]
):
  t = (p[0] - p1[0])/(p2[0] - p1[0])

  if t >= 0 and t <= 1:
    return True

  return False

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

class CsRbfWithFaults:
  def __init__(
    self, 
    points: np.ndarray[Any, np.dtype[np.float64]],
    faults: list[tuple[tuple[float, float], tuple[float, float]]],
    rbf: Callable[[float, float], float] = basis,
    epsilon: float = 1e-4,
    rs: float = 0.8
  ):
    n = len(points)

    tree = KDTree(points[:, :2], copy_data=True)

    max_count = 0
    min_count = n

    new_points = []

    rf = 2 * rs / math.sqrt(3)

    for p in points:    
      count = tree.query_ball_point(p[:2], rs, return_length=True)

      if count >= 2:
        max_count = max(max_count, count)
        min_count = min(min_count, count)

        new_points.append(p)

    points = np.array(new_points)

    tree = KDTree(points[:, :2], copy_data=True)

    res = []

    for p in points:
      found = tree.query_ball_point(p[:2], rs)

      res.append(set(found))

    faults_exclude_points(faults, tree, rf, points, res, rs, rs * epsilon)

    max_count = 0
    min_count = n
    new_points = []

    n = len(points)

    for i in range(0, n):
      s = res[i]

      if len(s) > 1:
        min_count = min(min_count, len(s))
        max_count = max(max_count, len(s))
        new_points.append(points[i])

    points = np.array(new_points)

    tree = KDTree(points[:, :2], copy_data=True)

    res = []
    fb = []

    n = len(points)

    for p in points:
      fb.append(p[2])

      found = tree.query_ball_point(p[:2], rs)

      res.append(set(found))

    faults_exclude_points(faults, tree, rf, points, res, rs, rs * epsilon)
    
    def mv(v):
      f = np.zeros(n)

      for i in range(n):
        for j in res[i]:
          a = rbf(
            distance(points[i], points[j]),
            rs
          )

          f[i] += a * v[j]

      return f

    A = LinearOperator((n, n), matvec=mv)    

    b = cg(A, fb)

    self.rs = rs
    self.epsilon = epsilon
    
    self.rbf = rbf

    self.b = b[0]
    self.points = points
    self.faults = faults

    self.tree = tree

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
          p = self.points[k]
          z[i, j] += self.b[k] * self.rbf(distance((x[i, j], y[i, j]), p), self.rs)

    return z

    
