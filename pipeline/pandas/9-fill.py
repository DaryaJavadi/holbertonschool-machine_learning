#!/usr/bin/env python3

"""
This module contains a function `fill` that cleans and fills missing
values in a DataFrame according to specified rules.
"""


def fill(df):
    """
    Remove the 'Weighted_Price' column, fill missing values in
    other columns as follows:
    - Close: forward fill (previous row)
    - High, Low, Open: fill with the corresponding Close value in the same row
    - Volume_(BTC) and Volume_(Currency): fill with 0

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the relevant columns.

    Returns:
    pd.DataFrame: The cleaned and filled DataFrame.
    """

    if 'Weighted_Price' in df.columns:
        df = df.drop(columns=['Weighted_Price'])

    df['Close'] = df['Close'].fillna(method='ffill')

    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])

    for col in ['Volume_(BTC)', 'Volume_(Currency)']:
        df[col] = df[col].fillna(0)

    return df
