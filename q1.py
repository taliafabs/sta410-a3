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


# TODO: (a) center and scale the columns of x so that they have mean 0 and standard deviation 1
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
# Step 1: Compute the singular values
singular_values = np.linalg.svd(X, compute_uv=False)
singular_values_normalized = np.linalg.svd(X_normalized, compute_uv=False)

# QR
np.linalg.qr(X)
np.linalg.qr(X_normalized)

# Compare the condition numbers before & after normalization
print(f"Un-normalized condition number: {unnormalized_cond_num}")
print(f"Normalized condition number: {normalized_cond_num}")

# Plot the singular values on a log scale
# un-normalized
plt.figure(figsize=(8, 5))
plt.semilogy(singular_values, marker='o', linestyle='-', markersize=3)
plt.title('Unnormalized singular values')
plt.xlabel('Index')
plt.ylabel('Singular value magnitude (log scale)')
plt.grid(True, which="both", ls='-', alpha=0.5)
# plt.show()
# normalized
plt.figure(figsize=(8, 5))
plt.semilogy(singular_values_normalized, marker='o', linestyle='-', markersize=3)
plt.title('Normalized singular values')
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
r_values = [1, 2, 5, 10, 15, 20, 25]

ratios = []

# un-normalized
for r in r_values:
    X_tilde = randomized_svd(X=X, r=r)
    numerator = np.linalg.norm((X - X_tilde), 1)
    denominator = np.linalg.norm(X, 1)
    ratio = numerator / denominator
    ratios.append(ratio)
print(ratios)

# normalized
ratios = []
for r in r_values:
    X_tilde = randomized_svd(X=X_normalized, r=r)
    numerator = np.linalg.norm((X_normalized - X_tilde), 1)
    denominator = np.linalg.norm(X_normalized, 1)
    ratio = numerator / denominator
    ratios.append(ratio)
print(ratios)
# ---------------------------------------------------------------------------------------------------------------------
