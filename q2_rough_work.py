import numpy as np
# Q2: Iterative Optimization

# TODO: 2(b)

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

def newton_raphson(alpha):
    x_0 = alpha
    pass