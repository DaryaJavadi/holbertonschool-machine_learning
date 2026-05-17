!/usr/bin/env python3
"""
module for task 4
"""

import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    calculates symmetric P affinities of data set
    """
    D, P, betas, H = P_init(X, perplexity)
    for i in range(X.shape[0]):
        Hi, Pi = HP(np.append(D[i, :i], D[i, i+1:]), betas[i])
        mini = None
        maxi = None
        Hdiff = Hi - H
        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                mini = betas[i].copy()
                if maxi is None:
                    betas[i] = betas[i] * 2
                else:
                    betas[i] = (betas[i] + maxi) / 2
            else:
                maxi = betas[i].copy()
                if mini is None:
                    betas[i] = betas[i] / 2
                else:
                    betas[i] = (betas[i] + mini) / 2
            Hi, Pi = HP(np.append(D[i, :i], D[i, i + 1:]), betas[i])
            Hdiff = Hi - H
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:X.shape[0]]))] = Pi
    return (P.T + P) / (2 * X.shape[0])
