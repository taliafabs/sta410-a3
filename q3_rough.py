import numpy as np

# TODO: 3(a) implement a function that given x returns f(x), the gradient, and the hessian

def func_a(x):
    x_1, x_2 = x[0], x[1]
    # compute f(x)
    f_x = (1-x_1)**2 + (x_2 - x_1**2)**2
    # compute the gradient
    # compute the hessian
    pass

# TODO: 3(b)