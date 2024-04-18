import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import arrow
import inspect

sig = inspect.signature (arrow)
  # get the list of parameter names from the signature object
params = list (sig.parameters.keys ())
  
# Example values for the 4x4 grid
values = np.random.rand(4, 4, 4)  # Replace with your values
#direction = max(values[0])
print(params)
# Create the 4x4 grid with values at different sides
plt.figure(figsize=(6, 6))
cell_size = 1
for row in range(4):
    for col in range(4):
        value_top = values[row, col, 0]
        value_right = values[row, col, 1]
        value_bottom = values[row, col, 2]
        value_left = values[row, col, 3]
        
        # Draw the grid cell
        plt.gca().add_patch(plt.Rectangle((col, 3 - row), cell_size, cell_size, fill=False, color='black'))
        # Place values at different sides
        plt.text(col + 0.5, 3 - row + 0.9, f'{value_top:0.1f}', va='center', ha='center', fontsize=value_top*7+5, color='black')
        plt.text(col + 0.9, 3 - row + 0.5, f'{value_right:0.1f}', va='center', ha='center', fontsize=value_right*7+5, color='black')
        plt.text(col + 0.5, 3 - row + 0.1, f'{value_bottom:0.1f}', va='center', ha='center', fontsize=value_bottom*7+5, color='black')
        plt.text(col + 0.1, 3 - row + 0.5, f'{value_left:0.1f}', va='center', ha='center', fontsize=value_left*7+5, color='black')
        
        if value_top > 0:
            arrow(col + 0.5, 3 - row + 0.5, 0, value_top * 0.2, width = value_top *0.03, color = 'green')
        else:
            arrow(col + 0.5, 3 - row + 0.5, 0, abs(value_top) * 0.2, width = abs(value_top) *0.03, color = 'red')
 
        if value_right > 0:
            arrow(col + 0.5, 3 - row + 0.5, value_right * 0.2, 0, width = value_right *0.03, color = 'green')
        else:
            arrow(col + 0.5, 3 - row + 0.5, abs(value_right) * 0.2, 0, width = abs(value_right) *0.03, color = 'red')
        
        if value_bottom > 0:
            arrow(col + 0.5, 3 - row + 0.5, 0, abs(value_bottom) * -0.2, width = abs(value_bottom) *0.03, color = 'green')
        else:
            arrow(col + 0.5, 3 - row + 0.5, 0, abs(value_bottom) * -0.2, width = abs(value_bottom) *0.03, color = 'red')

        if value_right > 0:
            plt.arrow(col + 0.5, 3 - row + 0.5, abs(value_left) * -0.2, 0, width = abs(value_left) *0.03, color = 'green')
        else:
            arrow(col + 0.5, 3 - row + 0.5, abs(value_left) * -0.2, 0, width = abs(value_left) *0.03, color = 'red')

# Customize plot appearance
plt.xlim(-0.5, 4.5)
plt.ylim(-0.5, 4.5)
plt.xticks([])
plt.yticks([])
plt.gca().invert_yaxis()
plt.grid()
plt.tight_layout()
# Show the plot
plt.show()