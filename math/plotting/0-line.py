#!/usr/bin/env python3
"""
0-line.py
This module contains a function `line` that plots a cubic line graph
from 0 to 10 using a solid red line.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plot a cubic function from 0 to 10 as a solid red line.
    The x-axis is explicitly set from 0 to 10.
    """
    x = np.arange(0, 11)
    y = x ** 3

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, color='red', linestyle='-')
    plt.xlim(0, 10)
    plt.show()
