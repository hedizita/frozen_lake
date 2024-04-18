import gymnasium as gym
import numpy as np
import random 
import torch
import torch.nn as nn
from torch.nn import Sequential, ReLU, Linear
import torch.optim as optim
import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.hidden1=Linear(2,128)
        self.relu=ReLU()
        self.hidden2=Linear(128,128)
        self.output=Linear(128,4)

    def forward(self, x):
        x = calculate_x_y(state)
        if isinstance(x[0], float):
            x = torch.tensor([x],dtype=torch.float)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        return x[0]


model = DQN()
optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.0)
loss_fn = nn.HuberLoss()

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

def calculate_x_y(state):
    x = state//n
    x = x/(n-1)
    y = state%n
    y = y/(n-1)
    return x, y


def greedy_policy(state):
    #print(model(state))
    #print(argmax(model(state)))
    action = random.choice(argmax(model(state)))
    return action

def epsilon_greedy_policy(state, epsilon):
    k = torch.rand(1)[0]
    print(k)
    if k > epsilon:
        action = random.choice(argmax(model(state)))
    else:
        action = random.choice([0,1,2,3])
    return action
        


def update_Q_network(state, action, next_rew, next_state):
    new_Q_value = next_rew + max(model(next_state))
    loss = loss_fn(model(state)[action], new_Q_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def visualization():
    #values = 
    print(values[1][1])
    plt.figure(figsize=(6, 6))
    cell_size = 1
    for row in range(4):
        for col in range(4):
            value_top = values[row, col, 0]
            value_right = values[row, col, 1]
            value_bottom = values[row, col, 2]
            value_left = values[row, col, 3]
            
            direction = np.argmax(values[row][col])
            k = 0
            j = 0
            
            if direction == 0:
                k = 0
                j = 1
            if direction == 1:
                k = 1
                j = 0
            if direction == 2:
                k = 0
                j = -1
            if direction == 3:
                k = -1
                j = 0

            plt.gca().add_patch(plt.Rectangle((col, 3 - row), cell_size, cell_size, fill=False, color='black'))

            plt.text(col + 0.5, 3 - row + 0.9, f'{value_top:0.1f}', va='center', ha='center', fontsize=value_top*7+5, color='black')
            plt.text(col + 0.9, 3 - row + 0.5, f'{value_right:0.1f}', va='center', ha='center', fontsize=value_right*7+5, color='black')
            plt.text(col + 0.5, 3 - row + 0.1, f'{value_bottom:0.1f}', va='center', ha='center', fontsize=value_bottom*7+5, color='black')
            plt.text(col + 0.1, 3 - row + 0.5, f'{value_left:0.1f}', va='center', ha='center', fontsize=value_left*7+5, color='black')

            plt.arrow(col + 0.5, 3 - row + 0.5, k * 0.3, j * 0.3, width = 0.03)
            
    plt.xlim(-0.5, 4.5)
    plt.ylim(-0.5, 4.5)
    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.grid()
    plt.tight_layout()
    plt.show()

#--------------------------------------
state,info = env.reset()
current_path = []
i = 0
while True:
    action = epsilon_greedy_policy(state, (1-i/1000))
    #visualization(action)
    next_state, next_rew, terminated, truncated, info = env.step(action)
    if terminated and next_rew != 1:
        next_rew = -1
    current_path.insert(0, [state, action, next_rew, next_state])

    for step in current_path:
        state, action, next_rew, next_state = step
        update_Q_network(state, action, next_rew, next_state)
        #print(next_rew)
    state = next_state

    
    if terminated or truncated:
        state,info = env.reset()
        i += 1
        current_path = []
