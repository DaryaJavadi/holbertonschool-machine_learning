#!/usr/bin/env python3

"""
This module contains a function `analyze` that computes descriptive
statistics for all columns except 'Timestamp'.
"""


def analyze(df):
    """
    Compute descriptive statistics for all columns except 'Timestamp'.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing a 'Timestamp' column.

    Returns:
    pd.DataFrame: A new DataFrame containing descriptive statistics.
    """
    df_numeric = df.drop(columns=['Timestamp'], errors='ignore')

    stats = df_numeric.describe()

    return stats
