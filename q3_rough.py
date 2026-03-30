import numpy as np


# TODO: 3(a) implement a function that given x returns f(x), the gradient, and the hessian

def func_a(x):
    """
    Given a 2x1 vector x, compute the value of the Rosenbrock function f(x) = (1 - x1)^2 + (x2 - x1^2)^2,
    its gradient vector f'(x), and its hessian matrix f''(x)
    """
    x_1, x_2 = x[0], x[1]
    # compute f(x)
    f_x = (1 - x_1) ** 2 + (x_2 - x_1 ** 2) ** 2
    # compute the gradient (2x1 vector)
    grad = np.array([-2 * (1 - x_1) - 4 * x_1 * (x_2 - x_1 ** 2),
                     2 * (x_2 - x_1 ** 2)])
    # compute the Hessian (2x2 matrix)
    hess = np.array([
        [12 * x_1 ** 2 - 4 * x_2 + 2, -4 * x_1],
        [-4 * x_1, 2]
    ])
    return f_x, grad, hess


# TODO: 3(b) implementation so far

def gradient_descent(x_0, alpha, epsilon, n_max=300):
    """
    Perform gradient descent with the following update rule at each iteration:
        x_k+1 = x_k - 𝛼 • ∇f(x_k)

    Terminate when || ∇f(x_k) || < ε or maximum number of iterations, nmax is reached.

    Return:
        - optimization path {x_k}
        - convergence history {f(x_k)}
    """
    x_k = x_0.copy()
    k = 0
    optimization_path = [x_k.copy()]
    convergence_history = [func_a(x_k)[0].copy()]
    while k < n_max:
        # func = func_a(x_k)
        # f_x_k, grad_x_k = func[0], func[1]
        # convergence_history.append(f_x_k.copy())
        grad_x_k = func_a(x_k)[1]
        # stop early if || ∇f(x_k) || < ε
        if np.linalg.norm(grad_x_k) < epsilon:
            return optimization_path, convergence_history
        # update rule: x_k+1 = x_k - 𝛼 • ∇f(x_k)
        x_k_1 = x_k - alpha * grad_x_k
        x_k = x_k_1
        # keep track of the gradient & value
        optimization_path.append(x_k.copy())
        convergence_history.append(func_a(x_k)[0].copy())
        k += 1
    return optimization_path, convergence_history

# try it out???
# how are x_0, alpha, epsilon chosen???
# in lec 8 we leaerned that epsilon should be very small

# TODO: 3(c)

# def phi_alpha(alpha, x_k):
#     grad_x_k = func_a(x_k)[1]
#     u_k = -1 * (grad_x_k / np.linalg.norm(grad_x_k))
