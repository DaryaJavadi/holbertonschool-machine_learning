#!/usr/bin/env python3
"""PCA function using numpy SVD"""
import numpy as np


def pca(X, var=0.95):
    """
    Function that performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) where:
           n is the number of data points
           d is the number of dimensions in each point
           all dimensions have a mean of 0 across all data points
        var: fraction of the variance that the PCA transformation
             should maintain

    Returns:
        W: weights matrix of shape (d, nd) where nd is the
           new dimensionality of the transformed X
    """
    _, s, vh = np.linalg.svd(X)

    variance = np.cumsum(s) / np.sum(s)

    r = np.argwhere(variance >= var)[0, 0]

    return vh[:r + 1].T
