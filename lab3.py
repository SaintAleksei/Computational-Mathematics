#!/usr/bin/python3

# Example of using interpolation API

import numpy as np
from sa_math import Interpolator, grid
from matplotlib import pyplot as plt
from scipy import interpolate 

def func(x):
  return x ** 2

def main(func, segment, sampling):
  # Creating grid from function
  x, y = grid(func, segment, sampling)

  # Creating some interpolators and plot each of them
  inter = Interpolator(x[::-1], y[::-1])
  newton = inter.build_Newton()
  lagrange = inter.build_Lagrange()
  splines = dict()
  splines[3] = inter.build_Splines(degree=3)

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

  # Some splines
  fig, ax = plt.subplots()
  ax.plot(x, y, 'ob')
  #ax.plot(x_func, splines[3](x_func), '-r', label='Кубические сплайны')
  scipy_spline = interpolate.CubicSpline(x, y)
  ax.plot(x_func, scipy_spline(x_func), '-b', label='Другие кубические сплайны')
  ax.legend(loc='lower right')

  plt.show()

if __name__ == '__main__':
  main(func, (0, 15), 10)
