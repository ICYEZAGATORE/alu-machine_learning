#!/usr/bin/env python3
"""Performs PCA on a dataset"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Parameters
    ----------
    X : numpy.ndarray
        shape (n, d), centered dataset
    var : float
        fraction of variance to preserve

    Returns
    -------
    W : numpy.ndarray
        shape (d, nd), principal components
    """
    n, d = X.shape

    # Compute covariance matrix
    cov = np.dot(X.T, X) / (n - 1)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvalues descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute cumulative variance ratio
    cum_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    # Select number of components to reach 'var'
    nd = np.searchsorted(cum_var, var) + 1

    # Return weight matrix
    W = eigenvectors[:, :nd]

    return W
