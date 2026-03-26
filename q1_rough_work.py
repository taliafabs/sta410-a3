import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels as sm
from statsmodels import datasets
import matplotlib.pyplot as plt

rng = np.random.default_rng(410)

# load the dataset (un-normalized)
X = sm.datasets.get_rdataset("mtcars").data.values
# X is a 32x11 matrix


# TODO: (a) center and scale the columns of x so that they have mean 0 and standard deviation 1
# can do this with sklearn
scaler = StandardScaler()
normalized_data_sklearn = scaler.fit_transform(X)
# or numpy (i think i will stick w numpy)
normalized_data_np = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X_normalized = normalized_data_np


# TODO: (b) how centering and scaling affects condition number
# Condition numbers
unnormalized_cond_num = np.linalg.cond(X)
normalized_cond_num = np.linalg.cond(X_normalized)
print(f"Un-normalized condition number: {unnormalized_cond_num}")
print(f"Normalized condition number: {normalized_cond_num}")

# QR factorization
np.linalg.qr(X)
np.linalg.qr(X_normalized)

# Plot the singular values on a log scale
# Step 1: Compute the singular values
singular_values = np.linalg.svd(X, compute_uv=False)
singular_values_normalized = np.linalg.svd(X_normalized, compute_uv=False)
# Step 2: Plot them on a log scale
# un-normalized
plt.figure(figsize=(8, 5))
plt.semilogy(singular_values, marker='o', linestyle='-', markersize=3)
plt.title('Unnormalized singular values')
plt.xlabel('Index')
plt.ylabel('Singular value magnitude (log scale)')
plt.grid(True, which="both", ls='-', alpha=0.5)
plt.show()
# normalized
plt.figure(figsize=(8, 5))
plt.semilogy(singular_values_normalized, marker='o', linestyle='-', markersize=3)
plt.title('Normalized singular values')
plt.xlabel('Index')
plt.ylabel('Singular value magnitude (log-scale)')
plt.grid(True, which="both", ls='-', alpha=0.5)
plt.show()


# TODO: (c)

# Step 1: generate random (p x r) matrix P with independent standard normal entries
p, r = X.shape[1], 5
P = np.zeros((p, r))
for i in range(p):
    for j in range(r):
        P[i][j] = rng.standard_normal()

# Step 2: compute Z= XP, which randomly samples the column space of X
# using r samples in the column space of p
Z = np.dot(X, P)

# Step 3: QR decomposition Z=QR, where Q is a (n x r) matrix of orthogonal columns and
# R is a (r x r) square matrix
# Q defines orthogonal basis for Z which approximates col space of X
Q, R = np.linalg.qr(Z)

# Step 4: project X onto orthoganal directions Q as Y = Q^T X
Y = np.dot(Q.T, X)

# Step 5:
# U_y, D_y, V_y_transpose = np.linalg.svd(Y)

# Step 6: approximate original matrix X
# X_approx =



