#!/usr/bin/env python3

"""
This module contains a function `array` that selects the last
10 rows of the 'High' and 'Close' columns from a pandas DataFrame
and converts them into a NumPy ndarray.
"""

import pandas as pd


def array(df):
    """
    Select the last 10 rows of the 'High' and 'Close' columns
    and convert them into a NumPy ndarray.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing 'High' and 'Close' columns.

    Returns:
    np.ndarray: A NumPy array of the selected values.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
