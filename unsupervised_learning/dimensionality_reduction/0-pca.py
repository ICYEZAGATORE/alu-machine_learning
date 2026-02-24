#!/usr/bin/env python3
"""PCA module"""


import numpy as np


def pca(X):
    """
    Performs Principal Component Analysis on a dataset

    Args:
        X (numpy.ndarray): shape (m, n) where:
            m = number of data points
            n = number of dimensions

    Returns:
        W (numpy.ndarray): matrix of principal components
    """
    # Compute covariance matrix
    cov = np.matmul(X.T, X) / (X.shape[0] - 1)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(cov)

    return U
