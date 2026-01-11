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
    """
    df = df.rename(columns={"Timestamp": "Datetime"})

    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")

    return df[["Datetime", "Close"]]
