import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
from matplotlib.pyplot import arrow

def draw_q_table(values, n):
    values = values.reshape((n, n, 4))
    fig = plt.figure(figsize=(10, 10))
    cell_size = 1
    maxval = np.max(np.abs(values))
    for row in range(n):
        for col in range(n):
            value_left = values[row, col, 0]
            value_top = values[row, col, 1]
            value_right = values[row, col, 2]
            value_bottom = values[row, col, 3]
            # Draw the grid cell
            plt.gca().add_patch(plt.Rectangle((col, row), cell_size, cell_size, fill=False, color='black'))
            # Place values at different sides
            plt.text(col + 0.5, row + 0.9, f'{value_top:0.2f}', va='center', ha='center', fontsize=10)
            plt.text(col + 0.82, row + 0.35, f'{value_right:0.2f}', va='center', ha='center', fontsize=10)
            plt.text(col + 0.5, row + 0.1, f'{value_bottom:0.2f}', va='center', ha='center', fontsize=10)
            plt.text(col + 0.17, row + 0.35, f'{value_left:0.2f}', va='center', ha='center', fontsize=10)

            # Normalization
            value_top = value_top/maxval
            value_bottom = value_bottom/maxval
            value_left = value_left/maxval
            value_right = value_right/maxval
            if value_top > 0:
                arrow(col + 0.5, row + 0.5, 0, abs(value_top) * 0.2, width = value_top *0.03, color = 'red')
            else:
                arrow(col + 0.5, row + 0.5, 0, abs(value_top) * 0.2, width = abs(value_top) *0.03, color = 'blue')
            if value_right > 0:
                arrow(col + 0.5, row + 0.5, abs(value_right) * 0.2, 0, width = value_right *0.03, color = 'red')
            else:
                arrow(col + 0.5, row + 0.5, abs(value_right) * 0.2, 0, width = abs(value_right) *0.03, color = 'blue')
            if value_bottom > 0:
                arrow(col + 0.5, row + 0.5, 0, abs(value_bottom) * -0.2, width = abs(value_bottom) *0.03, color = 'red')
            else:
                arrow(col + 0.5, row + 0.5, 0, abs(value_bottom) * -0.2, width = abs(value_bottom) *0.03, color = 'blue')
            if value_left > 0:
                plt.arrow(col + 0.5, row + 0.5, abs(value_left) * -0.2, 0, width = abs(value_left) *0.03, color = 'red')
            else:
                arrow(col + 0.5, row + 0.5, abs(value_left) * -0.2, 0, width = abs(value_left) *0.03, color = 'blue')
    # Customize plot appearance
    plt.xlim(-0.5, 0.5+n)
    plt.ylim(-0.5, 0.5+n)
    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.grid()
    plt.tight_layout()
    # Show the plot
    #plt.show()
    fig.canvas.draw()
    plt.close()
    img = np.asarray(fig.canvas.buffer_rgba())
    return img