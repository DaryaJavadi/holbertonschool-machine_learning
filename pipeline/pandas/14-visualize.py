#!/usr/bin/env python3

"""
This script processes a cryptocurrency DataFrame and visualizes daily
aggregated values starting from 2017.
"""

import pandas as pd
import matplotlib.pyplot as plt


def visualize(df):
    """
    Transform and visualize the DataFrame.

    Steps:
    - Remove 'Weighted_Price'
    - Rename 'Timestamp' to 'Date' and convert to datetime
    - Set 'Date' as index
    - Fill missing values as specified
    - Aggregate daily values from 2017 onwards
    - Return the transformed DataFrame
    """
    if 'Weighted_Price' in df.columns:
        df = df.drop(columns=['Weighted_Price'])

    if 'Timestamp' in df.columns:
        df = df.rename(columns={'Timestamp': 'Date'})

    df['Date'] = pd.to_datetime(df['Date'], unit='s')

    df = df.set_index('Date')

    df['Close'] = df['Close'].fillna(method='ffill')
    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])
    for col in ['Volume_(BTC)', 'Volume_(Currency)']:
        df[col] = df[col].fillna(0)

    df = df[df.index.year >= 2017]

    daily = df.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })

    daily.plot(y=['High', 'Low', 'Open', 'Close'], figsize=(12, 6), title='Daily Crypto Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    
    return daily
