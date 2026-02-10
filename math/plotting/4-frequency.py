#!/usr/bin/env python3
"""
4-frequency.py
Plot a histogram of student grades for Project A.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plots a histogram of student grades with specific bins and labels."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50).astype(int)

    plt.figure(figsize=(6.4, 4.8))
    bins = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')

    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(np.arange(0, 101, 10))
