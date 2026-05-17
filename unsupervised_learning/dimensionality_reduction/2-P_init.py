#!/usr/bin/env python3
"""Initializes variables for P affinities in t-SNE"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the P affinities in t-SNE.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        perplexity: the perplexity that all Gaussian distributions should have

    Returns:
        D: numpy.ndarray of shape (n, n) with squared pairwise distances
        P: numpy.ndarray of shape (n, n) initialized to 0s
        betas: numpy.ndarray of shape (n, 1) initialized to 1s
        H: Shannon entropy for the given perplexity
    """
    n, d = X.shape

    # Squared pairwise distances using broadcasting
    sum_X = np.sum(X ** 2, axis=1)
    D = sum_X + sum_X[:, np.newaxis] - 2 * np.dot(X, X.T)
    np.fill_diagonal(D, 0)

    # Initialize P to zeros
    P = np.zeros((n, n))

    # Initialize betas to ones (beta = 1 / (2 * sigma^2))
    betas = np.ones((n, 1))

    # Shannon entropy for given perplexity (base 2)
    H = np.log2(perplexity)

    return D, P, betas, H
