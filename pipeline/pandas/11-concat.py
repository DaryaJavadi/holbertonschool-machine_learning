#!/usr/bin/env python3

"""
This module contains a function `concat` that concatenates two DataFrames
after indexing on 'Timestamp' and selecting specific rows.
"""

import pandas as pd


def concat(df1, df2):
    """
    Index both DataFrames on 'Timestamp', select rows from df2 up to timestamp
    1417411920, concatenate them to the top of df1, and label the keys.

    Parameters:
    df1 (pd.DataFrame): First DataFrame (coinbase).
    df2 (pd.DataFrame): Second DataFrame (bitstamp).

    Returns:
    pd.DataFrame: Concatenated DataFrame with keys 'bitstamp' and 'coinbase'.
    """
    index = __import__('10-index').index

    df1_indexed = index(df1)
    df2_indexed = index(df2)

    df2_selected = df2_indexed.loc[:1417411920]

    concatenated = pd.concat(
        [df2_selected, df1_indexed],
        keys=['bitstamp', 'coinbase']
    )

    return concatenated
