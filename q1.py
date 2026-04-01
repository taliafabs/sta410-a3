import numpy as np
import statsmodels as sm
from statsmodels import datasets
import matplotlib.pyplot as plt

# QUESTION 1: RANDOMIZED SVD
# ---------------------------------------------------------------------------------------------------------------------

# random number generator
rng = np.random.default_rng(410)

# load the dataset, (n x p) matrix X, un-normalized
X_data = sm.datasets.get_rdataset("mtcars").data.values
n, p = X_data.shape[0], X_data.shape[1]

# (a) center and scale the columns of x so that they have mean 0 and standard deviation 1
# ---------------------------------------------------------------------------------------------------------------------
X_normalized = (X_data - np.mean(X_data, axis=0)) / np.std(X_data, axis=0)
# ---------------------------------------------------------------------------------------------------------------------


# (b) how centering & scaling affects condition numbers
# ---------------------------------------------------------------------------------------------------------------------
# Condition numbers
unnormalized_cond_num = np.linalg.cond(X_data)
normalized_cond_num = np.linalg.cond(X_normalized)

# Singular values
singular_values = np.linalg.svd(X_data, compute_uv=False)
singular_values_normalized = np.linalg.svd(X_normalized, compute_uv=False)

# QR
Q_uncentered, R_uncentered = np.linalg.qr(X_data)
Q_centered, R_centered = np.linalg.qr(X_normalized)

# Compare the condition numbers of X before & after normalization
print(f"Condition numbers of the matrix X:")
print(f"Before normalization (centering): {unnormalized_cond_num}")
print(f"After normalization (centering): {normalized_cond_num}")

# Condition numbers of R (should be the same as X)
print(f"Condition numbers of R from the QR factorization, X=QR")
print(f"Before centering & scaling: {np.linalg.cond(R_uncentered)}")
print(f"After centering & scaling: {np.linalg.cond(R_centered)}")

# Plot the singular values on a log scale

# before centering & scaling
plt.figure(figsize=(8, 5))
plt.semilogy(singular_values, marker='o', linestyle='-', markersize=3)
plt.title('Singular values before normalization')
plt.xlabel('Index')
plt.ylabel('Singular value magnitude (log scale)')
plt.grid(True, which="both", ls='-', alpha=0.5)
# plt.show()

# after centering & scaling
plt.figure(figsize=(8, 5))
plt.semilogy(singular_values_normalized, marker='o', linestyle='-', markersize=3)
plt.title('Singular values after normalization')
plt.xlabel('Index')
plt.ylabel('Singular value magnitude (log-scale)')
plt.grid(True, which="both", ls='-', alpha=0.5)
# plt.show()
# ---------------------------------------------------------------------------------------------------------------------


# (c) Randomized SVD Implementation
# ---------------------------------------------------------------------------------------------------------------------

def randomized_svd(X=X_normalized, r=5):
    """
    Randomized SVD Algorithm from the Assignment 3 handout, Steps 1 through 6
    """
    # step 1
    # (I initially wrote this part iteratively & asked chatgpt how to vectorize)
    P = rng.standard_normal(size=(p, r))
    # step 2
    Z = np.dot(X, P)
    # step 3
    Q, R = np.linalg.qr(Z)
    # step 4
    Y = np.dot(Q.T, X)
    # step 5
    U_y, svs, V_y_transpose = np.linalg.svd(Y, full_matrices=False)
    D_y = np.diag(svs)
    # step 6
    return Q @ U_y @ D_y @ V_y_transpose


# ---------------------------------------------------------------------------------------------------------------------


# (d)
# ---------------------------------------------------------------------------------------------------------------------
def find_smallest_ratio(X=X_normalized, threshold=0.2):
    """
    Find the smallest ratio of r to p for which:
        ||X - Q U_y D_y V_y^T||_1 / ||X||_1 < 0.2
    where ||A||_1 is the sum of the absolute values of matrix A.
    """
    p = X.shape[1]
    l1_norm_X = np.linalg.norm(X, ord=1)
    ratio_record=[]
    for r in range(1, p+1): # r = 1, 2, ... p
        X_r = randomized_svd(X=X, r=r)
        l1_norm_error = np.linalg.norm((X-X_r), ord=1)
        ratio = l1_norm_error / l1_norm_X
        ratio_record.append(ratio.copy())
        if ratio < threshold:
            print(f"r:{r}, p:{p}, smallest ratio of r to p:{r / p}, norm_ratio={ratio},")
            return ratio_record
    print(f"no r ≤ p achieved l1 norm ratio < {threshold}")
    return ratio_record

    # r = 1  # start with r=1
    # p = X.shape[1](X
    # norm_ratio = np.inf
    # while r < p and norm_ratio >= threshold:
    #     norm_ratio = np.linalg.norm((X - randomized_svd(X=X, r=r)), ord=1) / \
    #                  np.linalg.norm(X, ord=1)
    #     r += 1  # then increase it until threshold is met
    # if r <= p and norm_ratio < threshold:
    #     return print(f"r:{r}, p:{p}, smallest ratio of r to p:{r / p}, norm_ratio={norm_ratio},")
    # else:
    #     return print(f"no r ≤ p achieved < {threshold}")


print(f"Part (d) output:")
# before normalization aka centering & scaling
print(f"Without centering & scaling")
error_ratio_unnormalized = find_smallest_ratio(X=X_data)
print(error_ratio_unnormalized)
print(f"With centering & scaling")
# after normalization aka centering & scaling
error_ratio_normalized = find_smallest_ratio(X=X_normalized)
print(error_ratio_normalized)
# ---------------------------------------------------------------------------------------------------------------------
