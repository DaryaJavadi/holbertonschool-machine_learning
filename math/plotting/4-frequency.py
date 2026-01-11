#!/usr/bin/env python3
"""
4-frequency.py
This module contains a function `frequency` that plots a histogram
of student grades for Project A with exact binning and styling.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plot a histogram of student grades:
    - Bins every 10 units
    - Bars outlined in black
    - Labels and title set
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50).astype(int)
    plt.figure(figsize=(6.4, 4.8))

    bins = np.arange(0, 110, 10)

    plt.hist(student_grades, bins=bins, edgecolor='black')

    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.show()
