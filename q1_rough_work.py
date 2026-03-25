import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels as sm
from statsmodels import datasets

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

# condition numbers
unnormalized_cond_num = np.linalg.cond(X)
normalized_cond_num = np.linalg.cond(X_normalized)
print(f"Un-normalized condition number: {unnormalized_cond_num}")
print(f"Normalized condition number: {normalized_cond_num}")

# plot the singular values on a log scale





# TODO: (c)
r=5
