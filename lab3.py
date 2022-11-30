#!/usr/bin/python3

import numpy as np
from sa_math import Interpolator, grid
from matplotlib import pyplot as plt

def func(x):
  return x * np.sin(x)

segment = (0, 10)


def main():
  x, y = grid(func, segment, 10)
  print(x, y)

  inter = Interpolator(x, y)
  newton = inter.build_Newton()
  lagrange = inter.build_Lagrange()

  fig, ax = plt.subplots()
  ax.plot(x, y, 'ob')
  x_func = np.linspace(segment[0], segment[1], 1000)
  y_func = np.array([lagrange(v) for v in x_func])
  ax.plot(x_func, y_func, '-r', label='Полином Лагранжа')
  ax.legend()

  fig, ax = plt.subplots()
  ax.plot(x, y, 'ob')
  y_func = np.array([newton(v) for v in x_func])
  ax.plot(x_func, y_func, '-r', label='Полином Ньютона')
  ax.legend()

  plt.show()

if __name__ == '__main__':
  main()
