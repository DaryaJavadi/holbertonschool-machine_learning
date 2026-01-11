#!/usr/bin/env python3

"""
This module contains a function `high` that sorts a DataFrame
by the 'High' column in descending order.
"""


def high(df):
    """
    Sort the DataFrame by the 'High' column in descending order.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing a 'High' column.

    Returns:
    pd.DataFrame: The sorted DataFrame.
    """
    return df.sort_values('High', ascending=False)
