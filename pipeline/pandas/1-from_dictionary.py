#!/usr/bin/env python3

"""
This module creates a pandas DataFrame from a dictionary.

The DataFrame has:
- Column "First" with values 0.0, 0.5, 1.0, 1.5
- Column "Second" with values "one", "two", "three", "four"
- Row labels "A", "B", "C", "D"
"""

import pandas as pd

# Create the dictionary
data = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
}

# Define the row labels
index = ["A", "B", "C", "D"]

# Create the DataFrame
df = pd.DataFrame(data, index=index)
