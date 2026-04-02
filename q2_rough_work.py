import numpy as np
# Q2: Iterative Optimization

# TODO: 2(b)

def f_prime(x, alpha, gamma, lamb):
    if x > 0:
        return 2*x - 2*alpha + lamb*gamma*(x**(gamma-1))
    elif x < 0:
        return 2*x - 2*alpha - lamb*gamma*(x**(gamma-1))
    else:
        raise ValueError("Derivative undefined at x = 0")


def f_double_prime(x, alpha, gamma, lamb):
    if x > 0:
        return 2
    elif x < 0:
        return 2
    else:
        raise ValueError("Second derivative undefined at x=0")


def hess_f(x):
    pass

# i
alpha, gamma, lamb = 10, 0.5, 0.1
niter = 1000


def iterative_alg(alpha, gamma, lamb, niter):
    x_0 = alpha
    x_k = x_0
    k = 1
    while k <= niter:
        x_prev = x_k
        x_k = alpha - ((lamb * gamma)/2) * x_prev/(np.absolute(x_prev))**(gamma)




## ii: Newton-Raphson using x_0 = alpha

def newton_raphson(alpha, gamma, lamb, niter):
    x_0 = alpha
    x_k = x_0
    for k in range(1, niter+1):
        x_prev = x_k.copy()
        x_k = x_prev - (f_prime(x_prev, alpha, gamma, lamb))/(f_double_prime(x_prev, alpha, gamma, lamb))
