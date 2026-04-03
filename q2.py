import numpy as np
# Q2: ITERATIVE OPTIMIZATION METHODS


# (b) implement and compare two iterative algoirthms for computing x*
# --------------------------------------------------------------------------

# part i: iterative algorithm w/ update rule defined on handout
def iterative_alg(alpha, gamma, lamb, tolerance=1e-15, max_iter=5000):
    """
    Iterative root-finding algorithm with update rule:
        x_k = alpha - (lambda * gamma)/2 • (|x_{k-1}|^gamma)/x_{k-1}
    """
    x_0 = alpha  # set x_0=alpha
    converged = False
    x_history = [x_0]
    objective_history = [f_prime(x=x_0, alpha=alpha, gamma=gamma, lamb=lamb)]
    x_prev, k = x_0, 0
    while k < max_iter and not converged:
        # use update rule defined on a3 handout
        x_k = alpha - 0.5 * (lamb * gamma) * ((np.abs(x_prev)) ** gamma) / x_prev
        objective_k = f_prime(x=x_k, alpha=alpha, gamma=gamma, lamb=lamb)
        if np.abs(objective_k) < tolerance:
            converged = True
        x_history.append(x_k)
        objective_history.append(objective_k)
        x_prev = x_k
        k += 1
    return x_history, objective_history, k


# part ii: Newton-Raphson algorithm using initial condition x_0=alpha

# helpers to compute f'(x) and f"(x)
def f_prime(x, alpha, gamma, lamb):
    if x == 0:
        raise ValueError("f(x) is not differentiable at cusp x = 0")
    elif x > 0:
        return 2 * x - 2 * alpha + lamb * gamma * (np.abs(x)) ** (gamma - 1)
    else:  # x < 0
        return 2 * x - 2 * alpha - lamb * gamma * (np.abs(x)) ** (gamma - 1)


def f_double_prime(x, alpha, gamma, lamb):
    if x == 0:
        raise ValueError("cannot evaluate second derivative at cusp x=0")
    elif x > 0:
        second_deriv = 2 + lamb * gamma * (gamma - 1) * np.abs(x) ** (gamma - 2)
        # I got this next line from chatgpt to prevent
        # "RuntimeWarning: overflow encountered in scalar divide"
        return second_deriv if np.abs(second_deriv) > 1e-8 else np.sign(second_deriv) * 1e-8
    else:  # x < 0
        second_deriv = 2 - lamb * gamma * (gamma - 1) * np.abs(x) ** (gamma - 2)
        return second_deriv if np.abs(second_deriv) > 1e-8 else np.sign(second_deriv) * 1e-8


def newton_raphson(alpha, gamma, lamb, tolerance=1e-15, max_iter=5000):
    """
    The Newton-Raphson algorithm using initial condition x_0=alpha
    """
    x_0 = alpha
    x_history = [x_0]
    objective_history = [f_prime(x=x_0, alpha=alpha, gamma=gamma, lamb=lamb)]
    converged = False
    x_prev, k = x_0, 0
    while k < max_iter and not converged:
        # perform Newton-Raphson update
        derivative = objective_history[-1]
        second_derivative = f_double_prime(x=x_prev, alpha=alpha, gamma=gamma, lamb=lamb)
        x_k = x_prev - derivative / second_derivative
        objective_k = f_prime(x=x_k, alpha=alpha, gamma=gamma, lamb=lamb)
        if np.abs(objective_k) < tolerance:
            converged = True
        x_history.append(x_k)
        objective_history.append(objective_k)
        x_prev = x_k
        k += 1
    return x_history, objective_history, k


# use different values of (alpha, gamma, lamb) to test these algorithms &
#  determine which one is faster

# disclosure: i asked chat gpt to give me
# 3 sets of (alpha, gamma, lambda) to violate the constraint from part (a)

different_values = [(1, 0.5, 1), (2, 0.5, 1), (0.5, 0.8, 0.3), (5, 0.3, 0.5)]

for values in different_values:
    a, g, l = values[0], values[1], values[2]
    iter = iterative_alg(alpha=a, gamma=g, lamb=l)
    nr = newton_raphson(alpha=a, gamma=g, lamb=l)
    # print out the number of iterations required
    print(f"alpha={a}, gamma={g}, lambda={l}")
    print(f"number of iterations required for convergence:")
    print(f"iterative: {iter[2]}, newton-raphson: {nr[2]}")
    print(" ")
    # plot the convergence history
    # plt.figure(figsize=(10, 7))
    # plt.title("Iterative method (i) vs Newton-Raphson (ii) convergence")
    # plt.plot(iter[1])
    # plt.plot(iter[1])
    # plt.xlabel("# Iterations")
    # plt.ylabel("Value of the objective")
    # plt.show()

# # alpha=1.5, gamma=0.5, lamb=1,
# iter_results1 = iterative_alg(alpha=1.5, gamma=0.5, lamb=1)
# nr_results1 = newton_raphson(alpha=1.5, gamma=0.5, lamb=1)
#
# # alpha=0.5, gamma=0.8, lamb=0.3
# iter_results2 = iterative_alg(alpha=0.5, gamma=0.8, lamb=0.3)
# nr_results2 = newton_raphson(alpha=0.5, gamma=0.8, lamb=0.3)
#
# # alpha=2, gamma=0.6, lamb=1
# iter_results3 = iterative_alg(alpha=3, gamma=0.2, lamb=2)
# nr_results3 = newton_raphson(alpha=3, gamma=0.2, lamb=2)

