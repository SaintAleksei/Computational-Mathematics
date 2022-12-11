#!/usr/bin/python3

# Example of using SLAE

import sa_math as slae # Backward compatibility :)
from matplotlib import pyplot as plt
import numpy as np

def test_solver(solver, matrix, vector, solution, *, iters_list=range(0, 100, 10)):
  solution = np.array(solution, dtype=np.float64)

  slae_inst = slae.SLAE(matrix, vector)

  err = {
    slae.norm1: [],
    slae.norm2: [],
    slae.norm3: [],
  }
  for i in iters_list:
    for k, v in err.items():
      u_vector = solver(slae_inst, iters=i, converg=False)
      v.append(k(u_vector - solution))

  fig, ax = plt.subplots()
  ax.set_title(solver.__name__)
  ax.plot(iters_list, np.log(err[slae.norm1]), 'or', label='norm1')
  ax.plot(iters_list, np.log(err[slae.norm2]), 'og', label='norm2')
  ax.plot(iters_list, np.log(err[slae.norm3]), 'ob', label='norm3')
  ax.legend(loc='lower right')

def main():
  matrix =   [[100,  30, -70],\
              [ 15, -50,  -5],\
              [  6,   2,  20]]
  vector =    [ 60, -40,  28]
  solution =  slae.SLAE(matrix, vector).solve_Gaus()
  test_solver(slae.SLAE.solve_Jacobi, matrix, vector, solution) 
  test_solver(slae.SLAE.solve_Seidel, matrix, vector, solution,\
              iters_list=range(0, 50, 5))
  
  matrix   = [[1, 1],\
              [1, 2]]
  vector   =  [1, 1]
  solution =  slae.SLAE(matrix, vector).solve_Gaus()
  test_solver(slae.SLAE.solve_min_residual, matrix, vector, solution)
  test_solver(slae.SLAE.solve_steepest_descent, matrix, vector, solution)

  plt.show()

if __name__ == '__main__':
  main()
