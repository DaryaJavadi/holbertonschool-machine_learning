#!/usr/bin/env python3
"""Shannon entropy and P affinities for t-SNE"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point.

    Args:
        Di: numpy.ndarray of shape (n - 1,) containing pairwise distances
            between a data point and all other points except itself
        beta: numpy.ndarray of shape (1,) containing the beta value
              for the Gaussian distribution

    Returns:
        Hi: the Shannon entropy of the points
        Pi: numpy.ndarray of shape (n - 1,) containing the P affinities
    """
    # P affinities: exp(-||xi - xj||^2 * beta)
    Pi = np.exp(-Di * beta)

    # Normalize to get conditional probabilities
    Pi = Pi / np.sum(Pi)

    # Shannon entropy: H = -sum(Pi * log2(Pi))
    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
