#!/usr/bin/env python3
"""Test PCA implementation"""

import numpy as np
pca = __import__('0-pca').pca

np.random.seed(0)
a = np.random.normal(size=50)
b = np.random.normal(size=50)
c = np.random.normal(size=50)
d = 2 * a
e = -5 * b
f = 10 * c

# Build dataset
X = np.array([a, b, c, d, e, f]).T
m = X.shape[0]

# Center data
X_m = X - np.mean(X, axis=0)

# PCA default (var=0.95)
W = pca(X_m)
print(W)
print(W.shape)

# PCA with var=0.59
W2 = pca(X_m, var=0.59)
print(W2)
print(W2.shape)

# Optional: reconstruction error
T = np.matmul(X_m, W)
X_t = np.matmul(T, W.T)
print(np.sum(np.square(X_m - X_t)) / m)
