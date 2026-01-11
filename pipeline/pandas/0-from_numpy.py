#!/usr/bin/env python3

"""
This module contains a function `from_numpy` that converts
a NumPy ndarray into a pandas DataFrame with alphabetically
capitalized column names.
"""

import pandas as pd


def from_numpy(array):
    """
    Create a pandas DataFrame from a NumPy ndarray.

    Parameters:
    array (np.ndarray): The input array to convert.

    Returns:
    pd.DataFrame: A DataFrame with columns labeled A, B, C, ...
    """
    columns = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)
