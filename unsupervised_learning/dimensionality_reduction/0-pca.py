#!/usr/bin/env python3
"""Performs PCA on a dataset"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        Centered dataset
    var : float
        Fraction of variance to preserve

    Returns
    -------
    W : numpy.ndarray of shape (d, nd)
        Principal components
    """
    n, d = X.shape

    # Step 1: Singular Value Decomposition
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Step 2: Compute eigenvalues (variance along components)
    eigenvalues = (S ** 2) / (n - 1)

    # Step 3: Compute cumulative variance ratio
    cum_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    # Step 4: Select number of components to preserve 'var'
    nd = np.searchsorted(cum_var, var) + 1

    # Step 5: Return top components
    W = Vt.T[:, :nd]

    return W
