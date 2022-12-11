#!/usr/bin/python3

# Examples of using differentiator API

from sa_math import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

def func(x):
    return x * x * x * x + np.sin(x)

def func_d(x):
    return 4 * x * x * x + np.cos(x)

def func_dd(x):
    return 12 * x * x - np.sin(x)

dots_number = 10
x_start = 0.0
x_end   = 1000.0

def compute_error(data, model):
    return max(abs(data - model))

errors = {
    diff_1_1 : np.zeros(dots_number),
    diff_1_2 : np.zeros(dots_number),
    diff_2_2 : np.zeros(dots_number)
}

for i in range(0, dots_number):
    num = 2 ** (i + 2)
    x, y, y_d = diff_1_1(func, x_start, x_end, num)
    errors[diff_1_1][i] = compute_error(y_d, func_d(x))
    x, y, y_d = diff_1_2(func, x_start, x_end, num)
    errors[diff_1_2][i] = compute_error(y_d, func_d(x))
    x, y, y_dd = diff_2_2(func, x_start, x_end, num)
    errors[diff_2_2][i] = compute_error(y_dd, func_dd(x))

rg = range(2, dots_number + 2)
nums = 2 ** np.array(rg) 
x_values = np.log((x_end - x_start) / (nums + 1))

coeffs = {k: linregress(x_values, np.log(v)) for k, v in errors.items()}

plt.figure(num=1, figsize=(12,10))

plt.plot(x_values, np.log(errors[diff_1_1]), 'or', label=f"diff_1_1 ({coeffs[diff_1_1].slope:.3f})")
plt.plot(x_values, np.log(errors[diff_1_2]), 'og', label=f"diff_1_2 ({coeffs[diff_1_2].slope:.3f})")
plt.plot(x_values, np.log(errors[diff_2_2]), 'ob', label=f"diff_2_2 ({coeffs[diff_2_2].slope:.3f})")

x_approx = np.linspace(min(x_values), max(x_values), 1000)
plt.plot(x_approx, coeffs[diff_1_1].slope * x_approx + coeffs[diff_1_1].intercept, '-r')
plt.plot(x_approx, coeffs[diff_1_2].slope * x_approx + coeffs[diff_1_2].intercept, '-g')
plt.plot(x_approx, coeffs[diff_2_2].slope * x_approx + coeffs[diff_2_2].intercept, '-b')

plt.legend(loc='lower right', fontsize='xx-large')
plt.grid(visible=True)

plt.show()
