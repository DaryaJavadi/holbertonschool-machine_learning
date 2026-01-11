#!/usr/bin/env python3

"""
This module contains a function `prune` that removes any rows
where the 'Close' column has NaN values.
"""


def prune(df):
    """
    Remove rows where the 'Close' column has NaN values.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing a 'Close' column.

    Returns:
    pd.DataFrame: The DataFrame with NaN rows removed.
    """
    return df.dropna(subset=['Close'])
