#!/usr/bin/env python3
"""Performs PCA using SVD"""

import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on a dataset using SVD.

    Parameters
    ----------
    X : numpy.ndarray, shape (n, d)
        Centered dataset
    var : float
        Fraction of variance to preserve

    Returns
    -------
    W : numpy.ndarray, shape (d, nd)
        Principal components
    """
    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute explained variance ratio
    eigenvalues = S**2
    cum_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    # Select number of components to preserve `var`
    nd = np.searchsorted(cum_var, var) + 1

    # Return principal components
    W = Vt.T[:, :nd]

    return W
