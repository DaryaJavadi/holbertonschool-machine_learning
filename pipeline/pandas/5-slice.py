#!/usr/bin/env python3

"""
This module contains a function `slice` that extracts specific columns
and selects every 60th row from a pandas DataFrame.
"""


def slice(df):
    """
    Extract the columns High, Low, Close, and Volume_BTC,
    and select every 60th row.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the required columns.

    Returns:
    pd.DataFrame: The sliced DataFrame.
    """
    selected_cols = df[['High', 'Low', 'Close', 'Volume_(BTC)']]

    return selected_cols.iloc[::60]
