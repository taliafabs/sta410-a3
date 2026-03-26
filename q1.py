import numpy as np
import statsmodels as sm
from statsmodels import datasets
import matplotlib.pyplot as plt

# QUESTION 1: RANDOMIZED SVD
# ---------------------------------------------------------------------------------------------------------------------

# random number generator
rng = np.random.default_rng(410)


# load the dataset, (n x p) matrix X, un-normalized
X = sm.datasets.get_rdataset("mtcars").data.values
n, p = X.shape[0], X.shape[1]


# (a) center and scale the columns of x so that they have mean 0 and standard deviation 1
# ---------------------------------------------------------------------------------------------------------------------
normalized_data_np = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X_normalized = normalized_data_np
# ---------------------------------------------------------------------------------------------------------------------


# TODO: (b) how centering & scaling affects condition numbers
# ---------------------------------------------------------------------------------------------------------------------
# Condition numbers
unnormalized_cond_num = np.linalg.cond(X)
normalized_cond_num = np.linalg.cond(X_normalized)

# Singular values
singular_values = np.linalg.svd(X, compute_uv=False)
singular_values_normalized = np.linalg.svd(X_normalized, compute_uv=False)

# QR
np.linalg.qr(X)
np.linalg.qr(X_normalized)

# Compare the condition numbers before & after normalization
print(f"Condition numbers of the matrix X:")
print(f"Before normalization (centering): {unnormalized_cond_num}")
print(f"After normalization (centering): {normalized_cond_num}")

# Plot the singular values on a log scale
# un-normalized
plt.figure(figsize=(8, 5))
plt.semilogy(singular_values, marker='o', linestyle='-', markersize=3)
plt.title('Singular values before normalization')
plt.xlabel('Index')
plt.ylabel('Singular value magnitude (log scale)')
plt.grid(True, which="both", ls='-', alpha=0.5)
# plt.show()
# normalized
plt.figure(figsize=(8, 5))
plt.semilogy(singular_values_normalized, marker='o', linestyle='-', markersize=3)
plt.title('Singular values after normalization')
plt.xlabel('Index')
plt.ylabel('Singular value magnitude (log-scale)')
plt.grid(True, which="both", ls='-', alpha=0.5)
# plt.show()
# ---------------------------------------------------------------------------------------------------------------------


# TODO: (c)
# ---------------------------------------------------------------------------------------------------------------------
def randomized_svd(X=X_normalized, r=5):
    # step 1
    P = np.zeros((p, r))
    for i in range(p):
        for j in range(r):
            P[i][j] = rng.standard_normal()
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


# TODO: (d)
# ---------------------------------------------------------------------------------------------------------------------
def find_smallest_ratio(X = X_normalized):
    r, p = 1, X.shape[1]
    norm_ratio = np.inf
    while r < p and norm_ratio >= 0.2:
        norm_ratio = np.linalg.norm((X - randomized_svd(X=X, r=r)), ord=1) /\
                     np.linalg.norm(X, ord=1)
        r += 1
    if r <= p and norm_ratio < 0.2:
        return print(f"r:{r}, norm_ratio={norm_ratio}, smallest ratio:{r/p}")
    else:
        return print(f"no r<p achieved <0.2")


find_smallest_ratio(X=X)
find_smallest_ratio(X=X_normalized)
# r_values = [1, 2, 5, 10, 15, 20, 25]
#
# ratios = []
#
# # un-normalized
# for r in r_values:
#     X_tilde = randomized_svd(X=X, r=r)
#     numerator = np.linalg.norm((X - X_tilde), 1)
#     denominator = np.linalg.norm(X, 1)
#     ratio = numerator / denominator
#     ratios.append(ratio)
# print(ratios)

# normalized
# ratios = {}
# r = 1
# while r < p:
#     X_tilde = randomized_svd(X=X_normalized, r=r)
#     numerator = np.linalg.norm((X_normalized - X_tilde), ord=1)
#     denominator = np.linalg.norm(X_normalized, ord=1)
#     ratios[r] = (numerator / denominator)
#     r += 1
# print(ratios)
# X_tilde = randomized_svd(X=X_normalized, r=1)
# norm_ratio = np.linalg.norm((X_normalized - X_tilde), ord=1) / np.linalg.norm(X_normalized, ord=1)
# r = 2
# while r < p and norm_ratio >= 0.2:
#     X_tilde = randomized_svd(X=X_normalized, r=1)
#     norm_ratio = np.linalg.norm((X_normalized - X_tilde), ord=1) / np.linalg.norm(X_normalized, ord=1)

# ---------------------------------------------------------------------------------------------------------------------
