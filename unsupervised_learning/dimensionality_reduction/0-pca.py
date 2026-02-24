#!/usr/bin/env python3
"""PCA module"""


import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset

    Args:
        X (numpy.ndarray): shape (m, n)
        var (float): fraction of variance to preserve

    Returns:
        W (numpy.ndarray): matrix of principal components
    """
    # SVD
    U, S, Vt = np.linalg.svd(X)

    # Compute explained variance ratio
    explained = np.cumsum(S ** 2) / np.sum(S ** 2)

    # Find number of components to preserve `var`
    r = np.searchsorted(explained, var) + 1

    # Return first r principal components
    return Vt.T[:, :r]
