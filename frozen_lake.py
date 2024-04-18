import gymnasium as gym
import numpy as np
import random 
import matplotlib.pyplot as plt
import cv2
from matplotlib.pyplot import arrow

n = 8
env = gym.make(
        'FrozenLake-v1',
        is_slippery = False,
        render_mode = "human",
        map_name = f"{n}x{n}"
)

Q_table = np.zeros([n**2,4])

def argmax(x):
    maxidxs = []
    maxval = max(x)
    for i in range(len(x)):
        if x[i] == maxval:
            maxidxs.append(i)
    return maxidxs


def greedy_policy(state):
    action = random.choice(argmax(Q_table[state]))
    return action


def update_Q_table(state, action, next_rew, next_state):
    new_Q_value = next_rew + max(Q_table[next_state])
    #print(new_Q_value)
    Q_table[state][action] = new_Q_value

def draw_q_table(values):
    values = values.reshape((n, n, 4))
    print(values)
    fig = plt.figure(figsize=(6, 6))
    cell_size = 1
    for row in range(n):
        for col in range(n):
            value_left = values[row, col, 0]
            value_top = values[row, col, 1]
            value_right = values[row, col, 2]
            value_bottom = values[row, col, 3]

            # Normalization
            maxval = 0
            mag  = [] #size            
            mag.append(abs(value_top))
            mag.append(abs(value_right))
            mag.append(abs(value_bottom))
            mag.append(abs(value_left))

            maxval = max(mag)
            value_top = value_top/maxval
            value_bottom = value_bottom/maxval
            value_left = value_left/maxval
            value_right = value_right/maxval
            
            # Draw the grid cell
            plt.gca().add_patch(plt.Rectangle((col, row), cell_size, cell_size, fill=False, color='black'))
            # Place values at different sides
            plt.text(col + 0.5, row + 0.9, f'{value_top:0.1f}', va='center', ha='center', fontsize=abs(value_top)*7+(4/n)*5, color='black')
            plt.text(col + 0.9, row + 0.5, f'{value_right:0.1f}', va='center', ha='center', fontsize=abs(value_right)*7+(4/n)*5, color='black')
            plt.text(col + 0.5, row + 0.1, f'{value_bottom:0.1f}', va='center', ha='center', fontsize=abs(value_bottom)*7+(4/n)*5, color='black')
            plt.text(col + 0.1, row + 0.5, f'{value_left:0.1f}', va='center', ha='center', fontsize=abs(value_left)*7+(4/n)*5, color='black')
            
            if value_top > 0:
                arrow(col + 0.5, row + 0.5, 0, value_top * 0.2, width = value_top *0.03, color = 'green')
            else:
                arrow(col + 0.5, row + 0.5, 0, abs(value_top) * 0.2, width = abs(value_top) *0.03, color = 'blue')
    
            if value_right > 0:
                arrow(col + 0.5, row + 0.5, value_right * 0.2, 0, width = value_right *0.03, color = 'green')
            else:
                arrow(col + 0.5, row + 0.5, abs(value_right) * 0.2, 0, width = abs(value_right) *0.03, color = 'blue')
            
            if value_bottom > 0:
                arrow(col + 0.5, row + 0.5, 0, abs(value_bottom) * -0.2, width = abs(value_bottom) *0.03, color = 'green')
            else:
                arrow(col + 0.5, row + 0.5, 0, abs(value_bottom) * -0.2, width = abs(value_bottom) *0.03, color = 'blue')

            if value_right > 0:
                plt.arrow(col + 0.5, row + 0.5, abs(value_left) * -0.2, 0, width = abs(value_left) *0.03, color = 'green')
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

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return img

state,info = env.reset()
current_path = []
while True:
    action = greedy_policy(state)
    next_state, next_rew, terminated, truncated, info = env.step(action)
    if terminated and next_rew != 1:
        next_rew = -1
    current_path.insert(0, [state, action, next_rew, next_state])

    for step in current_path[:1]:
        state, action, next_rew, next_state = step
        update_Q_table(state, action, next_rew, next_state)
    state = next_state


    img = draw_q_table(Q_table)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    cv2.imshow('image', img)
    cv2.waitKey(1)


    if terminated or truncated:
        state,info = env.reset()
        current_path = []