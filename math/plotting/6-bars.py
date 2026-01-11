#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def bars():
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
