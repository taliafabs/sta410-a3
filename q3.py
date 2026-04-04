import numpy as np
import matplotlib.pyplot as plt


# QUESTION 3: GRADIENT DESCENT WITH STEP SIZE
# ---------------------------------------------------------------------------------------------------------------------


# (a) implement a function that given x returns f(x), the gradient, and the hessian
# ---------------------------------------------------------------------------------------------------------------------
def rosenbrock(x):
    """
    Given a 2x1 vector x, compute the value of the Rosenbrock function f(x) = (1 - x1)^2 + (x2 - x1^2)^2,
    its gradient vector, and its hessian matrix
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
# ---------------------------------------------------------------------------------------------------------------------


# (b) gradient descent implementation
# ---------------------------------------------------------------------------------------------------------------------
def rosenbrock_gradient_descent(x_0=np.array([-1.5, 1.1]), alpha=0.05, epsilon=1e-4, n_max=300):
    """
    Perform gradient descent with the following update rule at each iteration:
        x_k+1 = x_k - 𝛼 • ∇f(x_k)

    Terminate when || ∇f(x_k) || < ε or maximum number of iterations, nmax is reached.

    Return:
        - optimization path {x_k}
        - convergence history {f(x_k)}
    """
    optimization_path = [x_0.copy()]
    convergence_history = []
    x_k = x_0.copy()
    for k in range(n_max):
        f_x_k, grad_x_k, _ = rosenbrock(x_k)
        convergence_history.append(f_x_k.copy())
        # stop early if || ∇f(x_k) ||_2 < ε (I think this denotes L2 norm)
        if np.linalg.norm(grad_x_k) < epsilon:
            return optimization_path, convergence_history
        # otherwise perform a gradient descent update x_k+1 = x_k - 𝛼 • ∇f(x_k)
        x_k_1 = x_k - alpha * grad_x_k
        optimization_path.append(x_k_1.copy())
        x_k = x_k_1
    return optimization_path, convergence_history


def plot_rosenbrock_convergence_history(conv_hist, alpha, epsilon):
    """
    Plot the Rosenbrock Function gradient descent convergence history.
    Shows the value of the rosenbrock function f(x)=f(x) = (1 - x1)^2 + (x2 - x1^2)^2
    after every iteration of gradient descent.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(conv_hist, linewidth=2)
    plt.title(
        rf"Convergence History "
        rf"($\alpha={alpha}$)"
    )
    plt.xlabel("Iteration")
    plt.ylabel(r"f(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_rosenbrock_optimization_path(opt_path, alpha, epsilon):
    """
    Plot the Rosenbrock function contours & gradient descent optimization path.
    Disclosure of generative AI use: I used Copilot to produce this function. I asked it to
    help me plot the Rosenbrock function & optimization path.
    """
    path = np.array(opt_path)
    x1s, x2s = path[:, 0], path[:, 1]
    # Create a proper grid for contour plotting
    X1 = np.linspace(-2, 2, 400)
    X2 = np.linspace(-1, 3, 400)
    XX, YY = np.meshgrid(X1, X2)
    # Rosenbrock function for grid evaluation
    ZZ = (1 - XX) ** 2 + 100 * (YY - XX ** 2) ** 2
    plt.figure(figsize=(10, 7))
    # Log-spaced contour levels reveal the valley shape
    levels = np.logspace(-1, 3, 20)
    plt.contour(XX, YY, ZZ, levels=levels, cmap="viridis")
    # Optimization path
    plt.plot(x1s, x2s, "r.-", markersize=8, linewidth=1.5, label="Optimization path")
    # Start and end markers
    plt.plot(x1s[0], x2s[0], "go", label="Start")
    plt.plot(x1s[-1], x2s[-1], "bo", label="End")
    plt.title(
        rf"Optimization Path "
        rf"($\alpha={alpha}$, $\epsilon={epsilon}$)"
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()


# try out different values of 𝛼, using ε=0.00000001 ??? how was epsilon chosen???
learning_rates = [0.25, 0.1, 0.01, 0.001, 0.0001]
epsilons = [1e-4]
for lr in learning_rates:
    for eps in epsilons:
        rosenbrock_result = rosenbrock_gradient_descent(alpha=lr, epsilon=eps)
        plot_rosenbrock_optimization_path(opt_path=rosenbrock_result[0], alpha=lr, epsilon=eps)
        plot_rosenbrock_convergence_history(conv_hist=rosenbrock_result[1], alpha=lr, epsilon=eps)
# ---------------------------------------------------------------------------------------------------------------------


# (c) gradient descent improvement
# ---------------------------------------------------------------------------------------------------------------------
def phi_prime(alpha, x, u):
    """
    Computes phi'(alpha)
    """
    gradient = rosenbrock(x + alpha * u)[1]
    return np.dot(gradient.T, u)


def bisection_alpha(x, u, alpha_lb, alpha_ub, tau):
    """
    Bisection method to find a root for objective function ɸ'(𝛼)
    Statement on gen ai use: I used ChatGPT to debug this function.
    """
    phi_prime_lb = phi_prime(alpha_lb, x, u)
    phi_prime_ub = phi_prime(alpha_ub, x, u)
    count = 0
    # expanding the interval until it captures a root
    while phi_prime_lb * phi_prime_ub > 0 and count < 50:  # I got this suggestion from ChatGPT
        alpha_ub *= 2
        phi_prime_ub = phi_prime(alpha_ub, x, u)
        count += 1
    if (phi_prime_lb * phi_prime_ub) > 0:  # not able to capture a root after 50 iterations
        raise ValueError("Failed to capture a root within the interval")
    while True:
        alpha_mid = alpha_mid = 0.5 * (alpha_lb + alpha_ub)
        phi_prime_mid = phi_prime(alpha_mid, x, u)
        if abs(phi_prime_mid) < tau:
            return alpha_mid
        if phi_prime_lb * phi_prime_mid < 0:  # sign(phi_prime_lb]) ≠ sign(phi_prime_mid)
            alpha_ub = alpha_mid
            phi_prime_ub = phi_prime_mid
        else:  # sign(phi_prime_mid) ≠ sign(phi_prime_ub)
            alpha_lb = alpha_mid
            phi_prime_lb = phi_prime_mid

    # assert phi_prime_lb * phi_prime_ub < 0
    # alpha_mid = (alpha_lb + alpha_ub) / 2
    # # update rule
    # # compute a midpoint
    # phi_prime_mid = phi_prime(alpha_mid, x, u)
    # # if |phi_prime_mid| < tau, for some tolerance tau, return it!
    # if abs(phi_prime_mid) < tau:
    #     return alpha_mid
    # # otherwise update the search interval using alpha_mid
    # if (phi_prime_lb * phi_prime_mid) < 0:  # sign(phi_prime_lb]) ≠ sign(phi_prime_mid)
    #     return bisection_alpha(x, u, alpha_lb, alpha_mid, tau)
    # else:   # sign(phi_prime_mid) ≠ sign(phi_prime_ub)
    #     return bisection_alpha(x, u, alpha_mid, alpha_ub, tau)


def improved_rosenbrock_gradient_descent(x_0=np.array([-1.5, 1.1]), nmax=300, epsilon=1e-4):
    """
    Perform an improved gradient descent method with update rule:
        x_k+1 = x_l - alpha_k • ∇f(x_k)
    where step size alpha_k is selected using a bisection line search at
    each iteration.

    Return:
        - optimization path {x_k}
        - convergence history {f(x_k)}
    """
    opt_path = [x_0.copy()]
    conv_hist = []
    x_k = x_0.copy()
    for k in range(nmax):
        f_x_k, grad_x_k, hess_x_k = rosenbrock(x_k)
        conv_hist.append(f_x_k.copy())
        # stop early if || ∇f(x_k) ||_2 < ε (I think this denotes L2 norm)
        if np.linalg.norm(grad_x_k) < epsilon:
            return opt_path, conv_hist
        # compute the descent direction u_k
        u_k = -1 * (grad_x_k / np.linalg.norm(grad_x_k, ord=2))
        # compute the step size at the kth iteration
        # compute alpha_ub
        hess_eigvals = np.linalg.eigvals(hess_x_k)
        lambda_1_x = np.max(hess_eigvals)
        # upper bound for alpha: alpha <= 2 / lambda_1_x (lecture 8)
        alpha_k = bisection_alpha(x=x_k, u=u_k, alpha_lb=0,
                                  alpha_ub=(2 / lambda_1_x), tau=0.0001)
        # perform gradient descent update
        # x_k_1 = x_k - alpha_k * grad_x_k
        x_k_1 = x_k + alpha_k * u_k
        opt_path.append(x_k_1.copy())
        x_k = x_k_1
    return opt_path, conv_hist

# perform gd w/ adaptive step size
# improved_result = improved_rosenbrock_gradient_descent()

# # plot the optimization path (i used chat gpt to get a nice plot for this)
# path = np.array(improved_result[0])
# x1s, x2s = path[:, 0], path[:, 1]
# X1 = np.linspace(-2, 2, 400)
# X2 = np.linspace(-1, 3, 400)
# XX, YY = np.meshgrid(X1, X2)
# ZZ = (1 - XX) ** 2 + 100 * (YY - XX ** 2) ** 2
# plt.figure(figsize=(10, 7))
# levels = np.logspace(-1, 3, 20)
# plt.contour(XX, YY, ZZ, levels=levels, cmap="viridis")
# plt.plot(x1s, x2s, "r.-", markersize=8, linewidth=1.5, label="Optimization path")
# plt.plot(x1s[0], x2s[0], "go", label="Start")
# plt.plot(x1s[-1], x2s[-1], "bo", label="End")
# plt.title(
#     rf"Optimization path of gradient descent using adaptive step size "
#     rf"($\epsilon={1e-4}$)"
# )
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # plot the convergence history
# plt.figure(figsize=(8, 5))
# plt.plot(improved_result[1], linewidth=2)
# plt.title(
#     rf"Convergence history of gradient descent using adaptive step size "
#     rf"($\epsilon={1e-4})$"
# )
# plt.xlabel("Iteration")
# plt.ylabel(r"f(x)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# ---------------------------------------------------------------------------------------------------------------------
