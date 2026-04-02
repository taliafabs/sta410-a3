import numpy as np
from numpy import random
import matplotlib.pyplot as plt

rng = random.default_rng(410)

# QUESTION 4: EM ALGORITHM FOR A MIXTURE OF EXPONENTIALS
# ------------------------------------------------------


# TODO: (d) implement the EM algorithm. generate N=200 observations using true parameters
#  lambda1=1, lambda2=5, theta=0,4
# -------------------------------------------------------------------------------------------
# generate data
num_obs = 200
true_theta = 0.4
true_lambda1 = 1
true_lambda2 = 5
component = np.random.choice([1, 2], size=num_obs, p=[true_theta, 1 - true_theta])
observed_data = np.empty(num_obs)
observed_data[component == 1] = rng.exponential(scale=1 / true_lambda1, size=(component == 1).sum())
observed_data[component == 2] = rng.exponential(scale=1 / true_lambda2, size=(component == 2).sum())

# plot the observed data (just for visuals!)
plt.hist(observed_data, bins=50, density=True, alpha=0.7, color="orchid")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Two-component mixture of exponential distributions")
plt.show()


# implement the EM algorithm

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
    ll_init = 0     # TODO: implement log likelihood properly
    theta, lamb1, lamb2, ll = theta_init, lamb1_init, lamb2_init, ll_init
    log_likelihood_values = [ll_init]
    converged = False
    while not converged:     # run algorithm until convergence
        # E-step: compute E[Delta_i | X_i = x_i]
        expected_membership = (theta * lamb1 * np.exp(- lamb2 * x)) / \
                              (theta * lamb1 * np.exp(- lamb2 * x) + (1 - theta) * lamb1 * np.exp(- lamb2 * x))
        # M-step: compute updated lamb1, lamb2, theta using closed-form equations
        # TODO: compute the log likelihood at each iteration?????
        theta_new = (1/n) * np.sum(expected_membership)
        # TODO: derive closed form update eqns for lamb1, lamb2
        # TODO: implement updates for lamb1, lamb2
        lamb1_new = lamb1
        lamb2_new = lamb2
        ll_new = 0  # TODO: compute log likelihood properly! HERE!
        # TODO: correct convergence to be if log likelihood has converged since that's what we're maximizing
        if np.abs(ll_new - ll) < tolerance:
            converged = True
        # if np.abs(lamb1_new - lamb1) < tolerance and np.abs(lamb2_new - lamb2) < tolerance:
        #     converged = True
        # otherwise complete the update + keep track
        theta, lamb1, lamb2 = theta_new, lamb1_new, lamb2_new
        ll = ll_new
        log_likelihood_values.append(ll)
    return log_likelihood_values


# TODO: (e) plot the log likelihood at each iteration of the EM alg
#  illustrate monotone increasing property of EM
#  how many iterations until convergence?
# ------------------------------------------------------------------

# log_likelihood = EM()



# TODO: (f)


