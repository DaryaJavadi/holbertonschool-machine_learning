#!/usr/bin/env python3
"""
6-bars.py
This module contains a function `bars` that plots a stacked bar graph
showing the quantity of different fruits (apples, bananas, oranges, peaches)
for three people (Farrah, Fred, Felicia).
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plot a stacked bar graph of fruit quantities per person.

    - Apples are red
    - Bananas are yellow
    - Oranges are orange (#ff8000)
    - Peaches are peach (#ffe5b4)
    - Bars are stacked in order: apples, bananas, oranges, peaches
    - Bars width: 0.5
    - Y-axis label: 'Quantity of Fruit'
    - Y-axis range: 0 to 80 with ticks every 10 units
    - Title: 'Number of Fruit per Person'
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    rows = ('apples', 'bananas', 'oranges', 'peaches')
    columns = ('Farrah', 'Fred', 'Felicia')
    colors = ('red', 'yellow', '#ff8000', '#ffe5b4')

    bar_width = 0.5
    y_offset = np.zeros(len(columns))

    for i in range(len(fruit)):
        plt.bar(columns, fruit[i], bar_width, bottom=y_offset,
                color=colors[i], label=rows[i])
        y_offset += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title("Number of Fruit per Person")
    plt.legend()
    plt.show()
