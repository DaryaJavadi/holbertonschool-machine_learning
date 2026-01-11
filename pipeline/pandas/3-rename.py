#!/usr/bin/env python3

"""
This module contains a function `rename` that modifies a pandas
DataFrame by renaming the Timestamp column to Datetime, converting
it to datetime values, and displaying only the Datetime and Close columns.
"""

import pandas as pd


def rename(df):
    """
    Rename the Timestamp column to Datetime, convert it to datetime
    values, and keep only the Datetime and Close columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing a 'Timestamp' column.

    Returns:
    pd.DataFrame: Modified DataFrame with 'Datetime' and 'Close' columns.
    """
    # Rename the column
    df = df.rename(columns={"Timestamp": "Datetime"})

    # Convert to datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Keep only Datetime and Close columns
    return df[["Datetime", "Close"]]
