#!/usr/bin/python3

from sa_math import Equation

def square_matrix(x1, x2):
  return x1 * x2 + 2, x1 + x2 + 1

def square_matrix_deriv(x1, x2):
  return [x2, x1], [1, 1]

def square(x):
  return x ** 2 - x - 2

def square_deriv(x):
  return [2 * x - 1]

if __name__ == '__main__':
  eq = Equation(square_matrix, square_matrix_deriv, 2)
  print(eq.solve_relaxation(coeff=0.1))
  print(eq.solve_Newton(start=(0, 1)))

  eq = Equation(square, square_deriv, 1)
  print(eq.solve_relaxation(coeff=0.1))
  print(eq.solve_Newton(start=3))
  print(eq.solve_bisection(segment=(1, 3)))
