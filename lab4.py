#!/usr/bin/python3

from sa_math import Equation, norm1
import numpy as np
from matplotlib import pyplot as plt

# Roots: 2, -1

def square_matrix(x1, x2):
  return x1 * x2 + 2, x1 + x2 + 1

def square_matrix_deriv(x1, x2):
  return [x2, x1], [1, 1]

def square(x):
  return x ** 2 - x - 2

def square_deriv(x):
  return 2 * x - 1

# Plot error as function from iterations for simple equation solvers
def test_simple_solvers(func, deriv, solvers, dots=10, itdot=10):
  iters_list = [i * itdot for i in range(dots)]
  results = {k: np.zeros(dots) for k in solvers.keys()}

  eq = Equation(func, deriv, 1)

  for i in range(dots):
    for k, v in solvers.items():
      results[k][i] = k(eq, *v[0], iters=i * itdot, **v[1])

  fmt_list_ref = ['or', 'og', 'ob']
  fmt_list = fmt_list_ref.copy() 

  fig, ax = plt.subplots()
  ax.set_title('Results')
  for k, v in solvers.items():
    ax.plot(iters_list, results[k], fmt_list.pop(), label=str(k))
  ax.legend(loc='lower right')

  fig, ax = plt.subplots()
  fmt_list = fmt_list_ref.copy()
  ax.set_title('Errors')
  for k, v in solvers.items():
    ax.plot(iters_list, np.log(abs(results[k] - v[2])),\
            fmt_list.pop(), label=str(k))
  ax.legend(loc='upper right')

def test_system_solvers(func, deriv, nvars, solvers, norm=norm1, dots=10, itdot=10):
  iters_list = [i * itdot for i in range(dots)]
  results = {k: np.zeros(dots) for k in solvers.keys()}

  eq = Equation(func, deriv, nvars)

  for i in range(dots):
    for k, v in solvers.items():
      results[k][i] = norm(k(eq, *v[0], iters=i * itdot, **v[1]) - v[2])

  fmt_list_ref = ['or', 'og', 'ob']
  fmt_list = fmt_list_ref.copy() 

  fig, ax = plt.subplots()
  ax.set_title('Errors')
  for k, v in solvers.items():
    ax.plot(iters_list, np.log(results[k]), fmt_list.pop(), label=str(k))
  ax.legend(loc='upper right')


if __name__ == '__main__':
  matrix_solvers = {
    Equation.solve_relaxation: ((), {'coeff': 0.1}, (-1, 2)),
    Equation.solve_Newton: ((), {'start': (100, -100)}, (2, -1))
  }
  test_system_solvers(square_matrix, square_matrix_deriv, 2, matrix_solvers, itdot=1, dots=100)

  simple_solvers = {
    Equation.solve_secant: ((), {'x0': 10, 'x1': 15}, 2),
    Equation.solve_bisection: ((), {'segment': (1, 4)}, 2)
  }
  test_simple_solvers(square, square_deriv, simple_solvers, itdot=1, dots=100)

  plt.show()
