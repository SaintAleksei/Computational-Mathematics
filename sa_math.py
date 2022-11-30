import numpy as np

def norm_decorator(function):
  def norm_wrapper(obj):
    try:
      if type(obj) is not np.ndarray:
        raise TypeError(f'Object of type \'{np.ndarray}\' is required')
  
      if obj.ndim > 2 or obj.ndim < 1:
        raise ValueError('Bad object shape, N or (N, M) is required')

      return function(obj)
    except:
      print('Can\'t compute norm')
      raise

  return norm_wrapper

@norm_decorator
def norm1(obj):
  if obj.ndim == 1:
    return max(abs(obj))
  else:
    return max([sum(abs(row)) for row in obj])

@norm_decorator
def norm2(obj):
  if obj.ndim == 1:
    return sum(abs(obj))
  else:
    return max([sum(abs(row)) for row in obj.T])

@norm_decorator
def norm3(obj):
  if obj.ndim == 1:
    return np.sqrt(np.dot(obj, obj))
  else:
    return np.sqrt(max(np.linalg.eig(np.mul(obj.T, obj))[0]))

def grid(func, segment, sampling):
  x = np.linspace(segment[0], segment[1], sampling) 
  return x, func(x)

class Integrator:
  '''Integral calculation API'''
  pass

class Differentiator:
  pass

class Equation:
  pass

class DiffEquation:
  pass

class LagrangePolynomial:
  #TODO Add protection from different function values with the same argument
  '''Representation of Langrange interpolation polynomial'''
  def __init__(self, x, y):
    '''Initialize polynomial from given iterables'''
    try:
      x = np.array(x, dtype=np.float64) 
      y = np.array(y, dtype=np.float64)
      if x.ndim != y.ndim != 1 or x.size != y.size:
        raise ValueError('Bad x and y')
      self.__func_values = y
      self.__x_values = x
      
      self.__coeffs = np.full(self.__func_values.shape, 1, dtype=np.float64)
      for i in range(self.__coeffs.size):
        for j, v in enumerate(self.__x_values):
          if i != j:
            self.__coeffs[i] *= self.__x_values[i] - v
    except:
      print('Can\'t create Lagrange polynomial')
      raise

  def __setitem__(self, idx, y):
    '''Set function value at given index'''
    self.__func_values[idx] = float(y)

  def __getitem__(self, idx):
    '''Get xy pair from given index'''
    return self.__x_values[idx], self.__func_values[idx]

  def __call__(self, target):
    # TODO Add possibility use it with numpy arrays
    '''Compute polynomial value at given point (target)'''
    zero_idx = None
    target_mul = 1
    for i, v in enumerate(self.__x_values):
      if target - v == 0:
        zero_idx = i    
      else:
        target_mul *= target - v

    if zero_idx is not None:
      return self.__func_values[zero_idx]

    result = 0.0
    for x, f, c in zip(self.__x_values, self.__func_values, self.__coeffs):
      result += target_mul / (target - x) * f / c

    return result

class NewtonPolynomial:
  #TODO Add protection from different function values with the same argument
  '''Representation of Newton interpolation polynomial'''
  def __init__(self, x, y):
    '''Initialize polynomial from given iterables'''
    try:
      x = np.array(x, dtype=np.float64) 
      y = np.array(y, dtype=np.float64)
      if x.ndim != y.ndim != 1 or x.size != y.size != 0:
        raise ValueError('Bad x and y')

      combined = np.vstack((x, y)).T
      after_sort = np.sort(combined, axis=0)
      x = after_sort.T[0]
      y = after_sort.T[1]

      self.__x = x
      self.__y = y
      self.__sep_diffs = np.zeros((x.size, x.size))
      self.__sep_diffs[0] = np.copy(y)
      for i in range(1, x.size):
        for j in range(i, x.size):
          self.__sep_diffs[i, j] = self.__sep_diffs[i-1, j] -\
                                   self.__sep_diffs[i-1, j-1]
          self.__sep_diffs[i, j] /= self.__x[j] - self.__x[j-i]
      print(self.__sep_diffs)
    except:
      print('Can\'t create Newton polynomial')
      raise

  def __getitem__(self, idx):
    '''Get xy pair from given index'''
    return self.__x[idx], self.__y[idx]

  def __call__(self, target):
    '''Compute polynomial at given point (target)'''
    target_mul = self.__sep_diffs[0,0]
    result = target_mul
    for i in range(self.__x.size - 1):
      target_mul *= target - self.__x[i]
      result += target_mul * self.__sep_diffs[i+1,i+1]

    return result

  def append(self, xy_pair):
    '''Append new xy pair to the end, O(N)'''
    try: 
      if self.__x[self.__x.size-1] >= xy_pair[0]:
        raise ValueError('Appending makes x array unsorted')

      self.__x = np.append(self.__x, xy_pair[0])
      self.__y = np.append(self.__y, xy_pair[1])
      self.__sep_diffs = np.pad(self.__sep_diffs, ((0,1), (0,1)))
      self.__sep_diffs[0][self.__x.size - 1] = xy_pair[1]
      j = self.__sep_diffs.shape[1] - 1
      for i in range(1, self.__sep_diffs.shape[0]):
        self.__sep_diffs[i, j] = \
        self.__sep_diffs[i-1, j] - self.__sep_diffs[i-1,j-1]
        self.__sep_diffs[i, j] /= self.__x[j] - self._x[j-i]
    except:
      print('Can\'t append xy pair')
      raise

  def delete(self):
    '''Delete xy pair from the end'''
    try:
      if self.__x == 1:
        raise ValueError('Deletion of the last element')

      self.__x = np.delete(self.__x, self.__x.size - 1)
      self.__y = np.delete(self.__y, self.__y.size - 1)
      self.__sep_diffs = np.delete(\
        self.__sep_diffs,\
        self.__sep_diffs.shape[0] - 1,\
        axis = 0)
      self.__sep_diffs = np.delete(\
        self.__sep_diffs,\
        self.__sep_diffs.shape[1] - 1,\
        axis = 1)
    except:
      print('Can\'t delete xy pair')
      raise

class Interpolator:
  '''Interpolation API'''
  def __init__(self, xdata, ydata):
    '''Initialize Interpolator from given iterables'''
    try:
      xdata = np.array(xdata, dtype=np.float64)
      ydata = np.array(ydata, dtype=np.float64)
      if xdata.ndim != ydata.ndim != 1 or xdata.size != ydata.size:
        raise ValueError('Bad xdata and ydata')

      self.__x = xdata
      self.__y = ydata
    except:
      print(f'Can\'t create {type(self)}')
      raise

  def __setitem__(self, idx, xy_pair):
    '''Set xy pair'''
    try:
      if type(idx) is not int:
        raise TypeError(f'Bad type of idx \'{type(idx)}\'; \'{int}\' required')

      if idx >= self.__x.size:
        raise ValueError(f'Bad idx \'{idx}\'')

      self.__x[idx], self.__y[idx] = xy_pair[0], xy_pair[1]
    except:
      print(f'Can\'t set xy pair')
      raise

  def __getitem__(self, idx):
    '''Get xy pair'''
    try:
      if type(idx) is not int:
        raise TypeError(f'Bad type of idx \'{type(idx)}\'; \'{int}\' required')

      if idx >= self.__x.size:
        raise ValueError(f'Bad idx \'{idx}\'')

      return self.__x[idx], self.__y[idx]
    except:
      print('Can\'t get xy pair')
      raise

  def append(self, xy_pair):
    '''Append new xy pair'''
    try:
      self.__x = np.append(self.__x, xy_pair[0])
      self.__y = np.append(self.__y, xy_pair[0])
    except:
      print('Can\'t append value')
      raise

  def delete(self, idx):
    #TODO
    pass

  def build_Lagrange(self):
    '''Create instance of Lagrange polynomial'''
    return LagrangePolynomial(self.__x, self.__y)

  def build_Newton(self):
    '''Create instance of Newton polynomial'''
    return NewtonPolynomial(self.__x, self.__y)

class SLAE:
  '''System of Linear Algrebraic Equations solver'''

  def __init__(self, matrix, vector):

    '''Create a new instance from given input

    Arguments:
    matrix - Square matrix of SLAE. Should be something that can be pathed to numpy.array
    vector - Right side vector of SLAE. Should be something that can be pathed to numpy.array
    '''

    try:
      self.matrix = np.array(matrix, dtype=np.float64)
      self.vector = np.array(vector, dtype=np.float64)
    except:
      print('Can\'t create matrix and vector')
      raise

    if self.matrix.ndim != 2 or\
       self.matrix.shape[0] != self.matrix.shape[1] or\
       self.vector.ndim != 1 or\
       self.vector.shape[0] != self.matrix.shape[0]:
      raise ValueError(f'Bad shapes of matrix {self.matrix.shape}' +\
                       f'and vector {self.vector.shape}')

    try:
      self.det = np.linalg.det(self.matrix)
      if self.det == 0:
        raise ValueError('Degenerate matrix')
    except:
      print('Bad matrix')
      raise

  def condition_number(norm=norm1):

    '''Computing condition number of SLAE'''

    return norm(self.matrix) * norm(np.linalg.inv(self.matrix))

  def solve_Gaus(self, **kwargs):
    '''Solving SLAE by Gaus method'''
    try:
      matrix = np.copy(self.matrix)
      matrix = np.append(matrix,\
               np.reshape(self.vector, (self.vector.size, 1)), axis=1)
      bearings = [None] * (matrix.shape[1] - 1)

      for j in range(matrix.shape[1] - 1):
        for i, row in enumerate(matrix):
          if row[j] != 0.0 and i not in bearings:
            bearings[j] = i
            break
        matrix[bearings[j]] /= matrix[bearings[j], j]
        for i, row in enumerate(matrix):
          if i not in bearings:
            row -= row[j] * matrix[bearings[j]] 

      for j in reversed(bearings):
        for i, row in enumerate(matrix):
          if i not in bearings[j:]:
            row -= row[j] * matrix[bearings[j]]

      return matrix.T[matrix.shape[1] - 1]
    except:
      print('Can\'t solve SLAE by Gaus method')
      raise

  def solve_sweep(self):
    # TODO Doc-string

    try:
      # TODO
      pass
    except:
      print('Can\'t solve SLAE by Sweep method')
      raise

  def solve_Jacobi(self, **kwargs):

    '''Solve SLAE by Jacobi method

    Arguments:
    See SLAE.solve_iteration keyword arguments
    '''
    
    try:
      # Spliting matrix to inverse diagonal and zero-giagonal
      shape = self.matrix.shape
      diag_inv = np.full(shape, 0, dtype=np.float64)
      other= np.full(shape, 0, dtype=np.float64)
      for i in range(shape[0]):
        for j in range(shape[1]):
          if i == j:
            diag_inv[i, j] = self.matrix[i, j] ** -1
          else:
            other[i, j] = self.matrix[i, j]
      # Checking convergence
      # FIXME Maybe it can be ommited because solve_iteration do this
      #for i, v in enumerate(self.matrix):
      #  if v[i] < sum([val if j != i else 0.0 for j, val in enumerate(v)]):
      #    raise ValueError('Convergence condition isn\'t met')
      # Creating G and g for simple iteration method
      G_matrix = -1 * np.matmul(diag_inv, other)
      g_vector = np.matmul(diag_inv, self.vector)
      # Solving SLAE by simple iteration method
      return self.solve_iteration(G_matrix, g_vector, **kwargs)
    except:
      print('Can\'t solve SLAE by Jacobi method')
      raise
    
  def solve_Seidel(self, **kwargs):

    '''Solve SLAE by Seidel method

    Arguments:
    See SLAE.solve_iteration keyword arguments
    '''
    
    try:
      # Spliting matrix to diagonal, upper triangular and lower triangular
      shape = self.matrix.shape
      diag_lower_triangular = np.full(shape, 0, dtype=np.float64)
      upper_triangular = np.full(shape, 0, dtype=np.float64)
      for i in range(shape[0]):
        for j in range(shape[1]):
          if i <= j:
            diag_lower_triangular[i, j] = self.matrix[i, j]
          else:
            upper_triangular[i, j] = self.matrix[i, j]
      inverse = np.linalg.inv(diag_lower_triangular)
      # Checking convergence
      # TODO ?
      # Creating G and g for simple iteration method
      G_matrix = -1 * np.matmul(inverse, upper_triangular)
      g_vector = np.matmul(inverse, self.vector)
      # Solving SLAE by simple iteration method
      return self.solve_iteration(G_matrix, g_vector, **kwargs)
    except:
      print('Can\'t solve SLAE by Jacobi method')
      raise

  def solve_min_residual(self, **kwargs):

    '''Solving SLAE by Minimal Residual method

    Arguments:
    See slae.solve_func_minimize kwargs 
    '''

    def coeff_func(matrix, vector, residual):
      mat_residual = np.matmul(matrix, residual)
      numerator = np.dot(mat_residual, residual)
      denominator = 2 * np.dot(mat_residual, mat_residual)
      return numerator / denominator
    return self.solve_func_minimize(coeff_func, **kwargs)

  def solve_steepest_descent(self, **kwargs):

    '''Solving SLAE by Steepest Descent method

    Arguments:
    See slae.solve_func_minimize kwargs 
    '''

    def coeff_func(matrix, vector, residual):
      numerator = np.dot(residual, residual)
      denominator = 2 * np.dot(np.matmul(matrix, residual), residual)
      return numerator / denominator
    return self.solve_func_minimize(coeff_func, **kwargs)

  def solve_func_minimize(self, coeff_func, *, accuracy=1e-3,\
                          norm=norm1, u_start=None, iters=None, converg=True):
    
    '''Solving SLAE Au = f by minimization of functional F(u) = (Au, u) - 2(f, u)

    Arguments:
    coeff_func - function for computing a(k) vector, assumed a(k) = coeff_func(Au(k) - f)
    accuracy - Required computing accuracy, default is 1e-3
    norm - Norm to use, default is norm1
    u_start - Start value of u vector, default is (0, 0, ..., 0)
    converg - If true convergence condition is checked, default is True

    Assumed iteration: u(k+1) = u(k) - a(k) * grad(F)(u(k)) 
    '''

    try:
      # Input validation
      if not callable(coeff_func):
        raise TypeError('coeff_func should be callable')

      if type(iters) is not int:
        raise TypeError(f'iters of type \'{int}\' required')

      if u_start is None:
        u_start = np.full(self.vector.shape[0], 0.0, dtype=np.float64)

      # Check convergence condition
      if converg:
        if not np.all(np.linalg.eigvals(self.matrix) > 0):
          raise ValueError('Convergence condition isn\'t met')
        for i in range(self.matrix.shape[0]):
          for j in range(self.matrix.shape[1]):
            if self.matrix[i, j] != self.matrix[j, i]:
              raise ValueError('Convergence condition isn\'t met')
 
      # Iterating
      u_vector = u_start 
      residual = np.matmul(self.matrix, u_vector) - self.vector
      if iters is None:
        while norm(residual) < accuracy:
          coeff = coeff_func(self.matrix, self.vector, residual)
          u_vector = u_vector - 2 * coeff * residual
          residual = np.matmul(self.matrix, u_vector) - self.vector
        return u_vector
      else:
        for _ in range(iters):
          coeff = coeff_func(self.matrix, self.vector, residual)
          u_vector = u_vector - 2 * coeff * residual
          residual = np.matmul(self.matrix, u_vector) - self.vector
        return u_vector 
    except:
      print('Can\'t solve SLAE by functional minimization')
      raise

  @staticmethod
  def solve_iteration(G_matrix, g_vector, *, accuracy=1e-3, norm=norm1,\
                      iters=None, u_start=None, converg=True):

    '''Solving SLAE by Simple Iterations Method

    Arguments:
    G_matrix - G matrix
    g_vectotr - g vector
    accuracy - Required computing accuracy, default is 1e-3
    norm - Norm to use, default is norm1
    u_start - Start value of u vector, default is (0, 0, ..., 0)
    converg - If true convergence condition is checked, default is True

    Assumed iteration: u(k+1) = G * u(k) + g
    '''

    try:
      # Input validation
      if type(G_matrix) is not np.ndarray or type(g_vector) is not np.ndarray:
        raise TypeError(f'G_matrix and g_vector of type' +\
                        f'\'{np.ndarray}\' are required')

      if not callable(norm):
        raise TypeError(f'norm should be callable')

      if G_matrix.ndim != 2 or g_vector.ndim != 1 or\
         G_matrix.shape[0] != G_matrix.shape[1] or\
         G_matrix.shape[0] != g_vector.shape[0]:
        raise ValueError(f'Bad shapes of G matrix \'{G_matrix.shape}\'' +\
                         f'and g vector \'{g_vector.shape}\'')

      if u_start is not None:
        u_start = np.array(u_start)

      # Checking convergence condition
      if converg:
        G_norm = norm(G_matrix)
        if (G_norm > 1):
          raise ValueError('Convergence condition isn\'t met')

      # Iterating
      if u_start is not None and iters is None:
        u_vector = u_start
        u_vector_next = np.matmul(G_matrix, u_vector) + g_vector
        while norm(u_vector_next - u_vector) < accuracy:
          u_vector_next = np.matmul(G_matrix, u_vector) + g_vector
        return u_vector
      else:
        if u_start is None:
          u_start = np.full(g_vector.shape, 0.0, dtype=np.float64)
          # Computing required number of iterations
          if iters is None:
            g_norm = norm(g_vector)
            iters = np.log(accuracy * (1 - G_norm) / g_norm) / np.log(G_norm)
            iters = int(iters) + 1
        u_vector = u_start
        for _ in range(iters):
          u_vector = np.matmul(G_matrix, u_vector) + g_vector
        return u_vector
    except:
      print('Can\'t solve SLAE by iteration method')
      raise

# TODO Should be moved to Differentiator API

def compute_undef_coeffs(idx_start, idx_end, deriv_number):
    assert(isinstance(idx_start, int))
    assert(isinstance(idx_end, int))
    assert(isinstance(deriv_number, int))
    assert(idx_end >= 0 and idx_start >= 0 and deriv_number > 0)

    size = idx_end + idx_start + 1
    assert(size >= deriv_number + 1)

    # Solving linear system Ax = B

    B = np.zeros(size)
    B[deriv_number] = np.math.factorial(deriv_number)

    A = np.array([np.array(range(-1 * idx_start, idx_end + 1)) ** i for i in range(0, size)])

    return np.linalg.solve(A, B)

def __check_diff_args(func, x_start, x_end, num):
    assert(callable(func))
    assert(isinstance(x_start, float))
    assert(isinstance(x_end, float))
    assert(isinstance(num, int))

# The first derivative with the first power of approximation
def diff_1_1(func, x_start, x_end, num):
    __check_diff_args(func, x_start, x_end, num)
    assert(int(num) >= 2)
    assert(x_start < x_end)
    x_list = np.linspace(x_start, x_end, num)
    y_list = func(x_list)
    y2_list = np.concatenate((y_list[1:2], y_list[1:]))
    y1_list = np.concatenate((y_list[0:1], y_list[0:-1]))
    y_diff = y2_list - y1_list
    x_diff = (x_end - x_start) / (num - 1)
    return x_list, y_list, y_diff / x_diff

# The first derivative with the second power of approximation
def diff_1_2(func, x_start, x_end, num):
    __check_diff_args(func, x_start, x_end, num)
    assert(int(num) >= 3)
    assert(x_start < x_end)
    x_list = np.linspace(x_start, x_end, num)
    y_list = func(x_list)
    y_diff = y_list[2:] - y_list[0:-2]
    x_diff = (x_end - x_start) / (num - 1)
    diff_list =  y_diff / x_diff / 2
    diff_0 = np.dot(compute_undef_coeffs(0, 2, 1), y_list[0:3]) / x_diff
    diff_N = np.dot(compute_undef_coeffs(2, 0, 1), y_list[len(y_list)-3:]) / x_diff
    diff_list = np.insert(diff_list, 0, diff_0)
    diff_list = np.append(diff_list, diff_N)
    return x_list, y_list, diff_list

# The second derivative with the second power of approximation
def diff_2_2(func, x_start, x_end, num):
    __check_diff_args(func, x_start, x_end, num)
    assert(int(num) >= 4)
    x_list = np.linspace(x_start, x_end, num)
    y_list = func(x_list)
    y_diff = y_list[2:] + y_list[0:-2] - 2 * y_list[1:-1]
    x_diff = (x_end - x_start) / (num - 1)
    diff_list =  y_diff / (x_diff ** 2)
    diff_0 = np.dot(compute_undef_coeffs(0, 3, 2), y_list[0:4]) / (x_diff ** 2)
    diff_N = np.dot(compute_undef_coeffs(3, 0, 2), y_list[len(y_list)-4:]) / (x_diff ** 2)
    diff_list = np.insert(diff_list, 0, diff_0)
    diff_list = np.append(diff_list, diff_N)
    return x_list, y_list, diff_list
