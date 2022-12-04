#!/usr/bin/python3

import numpy as np
from sa_math import Interpolator, grid
from matplotlib import pyplot as plt

def func(x):
  return x**2 * np.sin(x)

def main(func, segment, sampling):
  # Creating grid from function
  x, y = grid(func, segment, sampling)

  # Creating some interpolators and plot each of them
  inter = Interpolator(x[::-1], y[::-1])
  newton = inter.build_Newton()
  lagrange = inter.build_Lagrange()
  splines = dict()
  splines[3] = inter.build_Splines(degree=3)
  splines[2] = inter.build_Splines(degree=2)

  # Lagrange's polynomial
  fig, ax = plt.subplots()
  ax.plot(x, y, 'ob')
  x_func = np.linspace(segment[0], segment[1], 1000)
  ax.plot(x_func, lagrange(x_func), '-r', label='Полином Лагранжа')
  ax.legend(loc='lower right')

  # Newton's polynomial
  fig, ax = plt.subplots()
  ax.plot(x, y, 'ob')
  y_func = newton(x_func)
  ax.plot(x_func, newton(x_func), '-r', label='Полином Ньютона')
  ax.legend(loc='lower right')
  fig, ax = plt.subplots()

  # Difference of Newton's and Lagrange's
  ax.plot(x_func, newton(x_func) - lagrange(x_func), 'ob',\
          label='Разница полиномов Ньютона и Лагранжа')
  ax.legend(loc='lower right')
  fig, ax = plt.subplots()

  # Some splines
  ax.plot(x, y, 'ob')
  ax.plot(x_func, splines[2](x_func), '-r', label='Квадратичные сплайны')
  ax.plot(x_func, splines[3](x_func), '-g', label='Кубические сплайны')
  ax.legend(loc='lower right')

  plt.show()

if __name__ == '__main__':
  main(func, (0, 5), 5)
