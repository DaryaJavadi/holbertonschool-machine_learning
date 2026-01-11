#!/usr/bin/env python3

"""
This module contains a function `hierarchy` that concatenates two DataFrames
(bitstamp and coinbase) with MultiIndex rearranged by Timestamp and
selected timestamp range.
"""

import pandas as pd


def hierarchy(df1, df2):
    """
    Rearrange the MultiIndex so Timestamp is the first level, 
    select rows from timestamps 1417411980 to 1417417980, 
    concatenate df2 on top of df1, add keys, and ensure chronological order.

    Parameters:
    df1 (pd.DataFrame): First DataFrame (coinbase).
    df2 (pd.DataFrame): Second DataFrame (bitstamp).

    Returns:
    pd.DataFrame: Concatenated DataFrame with keys 'bitstamp' and 'coinbase'
                  and Timestamp as the first level in chronological order.
    """
    index = __import__('10-index').index

    df1_indexed = index(df1)
    df2_indexed = index(df2)

    df2_selected = df2_indexed.loc[1417411980:1417417980]
    df1_selected = df1_indexed.loc[1417411980:1417417980]

    concatenated = pd.concat(
        [df2_selected, df1_selected],
        keys=['bitstamp', 'coinbase']
    )

    if isinstance(concatenated.index, pd.MultiIndex):
        concatenated = concatenated.reorder_levels([1, 0])
        concatenated = concatenated.sort_index()

    return concatenated
