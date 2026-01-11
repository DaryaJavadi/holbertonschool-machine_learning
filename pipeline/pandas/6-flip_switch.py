#!/usr/bin/env python3

"""
This module contains a function `flip_switch` that sorts a DataFrame
in reverse chronological order and then transposes it.
"""


def flip_switch(df):
    """
    Sort the DataFrame in reverse chronological order and transpose it.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a datetime index or column.

    Returns:
    pd.DataFrame: The transformed DataFrame.
    """
    sorted_df = df.sort_index(ascending=False)

    return sorted_df.T
