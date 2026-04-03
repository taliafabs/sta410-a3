import numpy as np
from numpy import random
import matplotlib.pyplot as plt

# from scipy import stats

rng = random.default_rng(410)


# trying this approach as well

# TODO: generate sample of N=200 observations from a two-component mixture of exponentials with
# true parameters lambda1=1, lambda2=5, theta=0.4
# 1. define pdf
def p(x, theta, lambda1, lambda2):
    """
    The 2-component exponential mixture pdf
    """
    return theta * lambda1 * np.exp(-lambda1 * x) + (1 - theta) * lambda2 * np.exp(-lambda2 * x)


# 2. generate observed data
num_obs = 200
true_theta = 0.4
true_lambda1 = 1
true_lambda2 = 5
components = np.random.choice([1, 2], size=num_obs, p=[true_theta, 1 - true_theta])
observed_data = np.empty(num_obs)
observed_data[components == 1] = rng.exponential(scale=1 / true_lambda1, size=(components == 1).sum())
observed_data[components == 2] = rng.exponential(scale=1 / true_lambda2, size=(components == 2).sum())

# 3. plot observed data pdf???
plt.hist(observed_data, bins=50, density=True, alpha=0.7, color="orchid")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Two-component mixture of exponential distributions")
plt.show()


# TODO: implement the EM alg

def EM(x=observed_data, tolerance=1e-6):
    """
    Expectation maximization algorithm for a mixture of two exponential components.
    """
    # initialize the parameters
    n = len(x)
    x1, x2 = x[component == 1], x[component == 2]
    theta_init = 0.5
    lamb1_init = np.mean(x1)
    lamb2_init = np.mean(x2)
    Delta = np.where(component == 1, 1, 0)
    # TODO: initialize the log likelihood using complete data log likelihood (i think)
    ll_complete = np.sum(Delta * (np.log(theta_init) + np.log(lamb1_init) - lamb1_init * x) +
                         (1 - Delta) * (np.log(1 - theta_init) + np.log(lamb2_init) - lamb2_init * x)
                         )
    # ll_new = np.sum(expected_membership) * np.log(theta_new) + \
    #          np.sum(1 - expected_membership) * np.log(1 - theta_new) + \
    #          np.sum(expected_membership * (np.log(lamb1_new) - lamb1_new * x)) + \
    #          np.sum((1 - expected_membership) * (np.log(lamb2_new) - lamb2_new * x))
    theta, lamb1, lamb2, ll = theta_init, lamb1_init, lamb2_init, ll_complete
    log_likelihood_values = [ll]
    converged = False
    while not converged:  # run algorithm until convergence
        # E-step: compute E[Delta_i | X_i = x_i]
        expected_membership = (theta * lamb1 * np.exp(- lamb1 * x)) / \
                              (theta * lamb1 * np.exp(- lamb1 * x) + (1 - theta) * lamb2 * np.exp(- lamb2 * x))
        # M-step: compute updated lamb1, lamb2, theta using closed-form equations
        # TODO: compute the log likelihood at each iteration?????
        theta_new = (1 / n) * np.sum(expected_membership)
        # TODO: derive closed form update eqns for lamb1, lamb2
        # TODO: implement updates for lamb1, lamb2
        lamb1_new = np.sum(expected_membership) / np.sum(expected_membership * x)
        lamb2_new = np.sum(1 - expected_membership) / np.sum((1 - expected_membership) * x)
        # TODO: update the observd log likelihood
        ll_new = np.sum(np.log(theta_new * lamb1_new * np.exp(- lamb1_new * x) +
                               (1 - theta_new) * lamb2_new * np.exp(- lamb2_new * x)
                               )
                        )
        # # expected log likelihood??? im pretty sure thats what this is
        # ll_new = np.sum(expected_membership) * np.log(theta_new) + \
        #          np.sum(1 - expected_membership) * np.log(1 - theta_new) + \
        #          np.sum(expected_membership * (np.log(lamb1_new) - lamb1_new * x)) + \
        #          np.sum((1 - expected_membership) * (np.log(lamb2_new) - lamb2_new * x))
        # TODO: correct convergence to be if log likelihood has converged since that's what we're maximizing
        if np.abs(ll_new - ll) < tolerance: # stop if the EM alg has converged
            converged = True
        # otherwise complete the update + keep track
        theta, lamb1, lamb2 = theta_new, lamb1_new, lamb2_new
        ll = ll_new
        log_likelihood_values.append(ll)
    return log_likelihood_values

# def EM(x, tolerance=1e-6):
#     # initialize the parameters
#     n = len(x)
#     theta_initial = 0.5
#     lambda1_initial = 1 / np.mean(x[components == 1])
#     lambda2_initial = 1 / np.mean(x[components == 2])
#     theta, lambda1, lambda2 = theta_initial, lambda1_initial, lambda2_initial
#     theta_history = []
#     lambda1_history = []
#     lambda2_history = []
#     while True:
#         # E-step:
#         expected_membership = (theta * lambda1 * np.exp(- lambda1 * x)) / \
#                               (theta * lambda1 * np.exp(- lambda1 * x) + (1 - theta) * lambda2 * np.exp(-lambda2 * x))
#         # M-step
#         theta_new = (1/n) * np.sum(expected_membership)
#         lambda1_new = lambda1 # TODO: fix these
#         lambda2_new = lambda2
#         converged = np.abs(lambda1_new - lambda1) < tolerance and \
#                     np.abs(lambda2_new - lambda2) < tolerance
#         if converged:
#             break
#         theta, lambda1, lambda2 = theta_new, lambda1_new, lambda2_new
#     return theta, lambda1, lambda2

# def EM(niter, x):
#     # initialize scale params
#     lambda1_initial = 1
#     lambda2_initial = 1
#     # initialize mixing weights
#     theta_initial = 0.5
#     lambda1_k, lambda2_k, theta_k = lambda1_initial, lambda2_initial, theta_initial
#     for k in range(1, niter+1):
#         x_k = x[k]
#         # E-step
#         d_i = (theta_k * lambda1_k * np.exp(-lambda1_k * x[k]))
#         # M-step
#


# -------------------------


# def EM(lambda1=1, lambda2=5, theta=0.4, n=200):
#     """
#     Perform the expectation maximization (EM) algorithm,
#     Generate 200 observations
#     """
#     converged = False
#     while not converged:
#         pass

# plot log likelihood at each iteration

# TODO: 4(c) implement the EM algorithm


# # define the true parameters
# n=200
# # scale parameters for the two exponentials
# lambda1, lambda2 = 1, 5
# # weight parameter
# theta=0.4
#
# # generate samples from each component using true params
# component1 = np.random.exponential(scale=lambda1, size=int(n * theta))
# component2 = np.random.exponential(scale=lambda1, size=int(n * (1 - theta)))
#
# # combine the data to get mixture of exponentials
# mixture_data = np.concatenate([component1, component2])
#
# # visualize
#
