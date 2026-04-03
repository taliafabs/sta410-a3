import numpy as np
from numpy import random
import matplotlib.pyplot as plt

rng = random.default_rng(123)

# QUESTION 4: EM ALGORITHM FOR A MIXTURE OF EXPONENTIALS
# ------------------------------------------------------


# TODO: (d) implement the EM algorithm. generate N=200 observations using true parameters
#  lambda1=1, lambda2=5, theta=0,4
# -------------------------------------------------------------------------------------------
# generate the data
num_obs = 200   # N=200 observations
true_theta, true_lambda1, true_lambda2 = 0.4, 1, 5  # true parameter values
component = rng.binomial(1, true_theta, size=num_obs)
observed_data = np.where(component == 1,
                         rng.exponential(scale=1/true_lambda1, size=num_obs),
                         rng.exponential(scale=1/true_lambda2, size=num_obs)
                         )

# plot the observed data
plt.hist(observed_data, bins=50, density=True, alpha=0.7, color="turquoise")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Two-component mixture of exponential distributions")
plt.show()


def EM(x=observed_data, theta_init=0.5, lamb1_init=2.0,
       lamb2_init=3.0, tolerance=1e-12):
    """
    Expectation maximization algorithm for a mixture of two exponential components.
    """
    n = len(x)
    # initialize & keep track of log likelihood
    ll_init = np.sum(np.log(theta_init * lamb1_init * np.exp(- lamb1_init * x) +
                            (1 - theta_init) * lamb2_init * np.exp(- lamb2_init * x)
                            )
                     )
    log_likelihood_values = [ll_init]
    theta, lamb1, lamb2, ll = theta_init, lamb1_init, lamb2_init, ll_init
    converged = False
    while not converged:  # run algorithm until convergence
        # E-step: compute E[Delta_i | X_i = x_i]
        expected_membership = (theta * lamb1 * np.exp(- lamb1 * x)) / \
                              (theta * lamb1 * np.exp(- lamb1 * x) + (1 - theta) * lamb2 * np.exp(- lamb2 * x))
        # M-step: compute updated lamb1, lamb2, theta using closed-form equations
        # to maximize expected log likelihood
        theta_new = (1 / n) * np.sum(expected_membership)
        lamb1_new = np.sum(expected_membership) / np.sum(expected_membership * x)
        lamb2_new = np.sum(1 - expected_membership) / np.sum((1 - expected_membership) * x)
        # update the observed data log likelihood
        ll_new = np.sum(np.log(theta_new * lamb1_new * np.exp(- lamb1_new * x) +
                               (1 - theta_new) * lamb2_new * np.exp(- lamb2_new * x)
                               )
                        )
        if np.abs(ll_new - ll) < tolerance:  # stop if the EM alg has converged
            converged = True
        # otherwise update params + keep track of log likelihood
        log_likelihood_values.append(ll_new)
        theta, lamb1, lamb2, ll = theta_new, lamb1_new, lamb2_new, ll_new
    return lamb1, lamb2, theta, log_likelihood_values


# TODO: (e) plot the log likelihood at each iteration of the EM alg
#  illustrate monotone increasing property of EM
#  how many iterations until convergence?
# ------------------------------------------------------------------

# run the EM algorithm on a 2 component mixture of exponentials
lamb1_hat, lamb2_hat, theta_hat, log_likelihood = EM()

# plot log likelihood at each iteration
plt.figure(figsize=(10, 7))
plt.title("Log Likelihood vs Number of EM Iterations")
plt.xlabel("# iterations")
plt.ylabel("Observed data log likelihood")
plt.plot(log_likelihood, '-o', markersize=2.1, color="magenta")
plt.show()

# determine how many iterations EM required before convergence
num_its_conv = len(log_likelihood)
print(f"Number of EM iterations required before convergence: {num_its_conv}")


# TODO: (f) Is the observed data log likelihood unimodal? Try at least 3 different starting values 
#  and coment on whether EM runs converge to the same solution
# -----------------------------------------

# starting values that are nowhere near true values
result1 = EM(x=observed_data, theta_init=0.95, lamb1_init=16, lamb2_init=1)
# starting values somewhat close
result2 = EM(x=observed_data, theta_init=0.45, lamb1_init=1.5, lamb2_init=4)
# another arbitrarily chosen starting point
result3 = EM(x=observed_data, theta_init=0.67, lamb1_init=6, lamb2_init=7)
# starting point very close to true values
result4 = EM(x=observed_data, theta_init=0.38, lamb1_init=0.98, lamb2_init=5.2)
result5 = EM(x=observed_data, theta_init=0.1, lamb1_init=10, lamb2_init=10)

print(f"EM solutions after convergence using different starting values")
print(" ")
print(f"starting values: lambda1=16, lambda2=1, theta=0.95")
print(f"lambda1:{result1[0]}, lambda2: {result1[1]}, theta: {result1[2]}")
print(" ")
print(f"starting values: lambda1=1.5, lambda2=4, theta=0.45")
print(f"lambda1:{result2[0]}, lambda2: {result2[1]}, theta: {result2[2]}")
print(" ")
print(f"starting values: lambda1=6, lambda2=7, theta=0.67")
print(f"lambda1:{result3[0]}, lambda2: {result3[1]}, theta: {result3[2]}")
print(" ")
print(f"starting values: lambda1=10, lambda2=10, theta=0.1")
print(f"lambda1:{result4[0]}, lambda2: {result4[1]}, theta: {result4[2]}")

