import gymnasium as gym
import numpy as np
import random 
import torch
import torch.nn as nn
from torch.nn import Sequential, ReLU, Linear
import torch.optim as optim
import visualize
import cv2
import wandb
from collections import defaultdict
cv2.namedWindow('image', cv2.WINDOW_NORMAL)


NUM_OF_EPISODES = 200
DISCOUNT = 0.9
N = 4
LR = 0.0005
EPSILON_INIT = 0.2
EPSILON_FINAL = 0.01
HIDDEN_WIDTH = 128
BATCH_SIZE = 32
MAX_NUM_STEPS = 50*N

wandb.init(
    project="FrozenLake",
    config={
        "num_of_episodes":NUM_OF_EPISODES,
        "discount":DISCOUNT,
        "map_size":N,
        "learning_rate":LR,
        "epsilon_init":EPSILON_INIT,
        "epsilon_final": EPSILON_FINAL,
        "hidden_width":HIDDEN_WIDTH,
        "max_num_steps": MAX_NUM_STEPS,
        "batch_size": BATCH_SIZE
    },
    entity="nagymaros2023"
)


#Calculate x and y coordinates
def calculate_x_y(state):
    x = state//N
    x = x/(N-1)
    y = state%N
    y = y/(N-1)
    return torch.tensor([x, y])

def calculate_onehot(state):
    ret = torch.zeros(N**2)
    ret[state]=1
    return ret
#Defining the DQN model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.hidden1=Linear(N**2,HIDDEN_WIDTH)
        self.relu=ReLU()
        self.hidden2=Linear(HIDDEN_WIDTH,HIDDEN_WIDTH)
        self.output=Linear(HIDDEN_WIDTH,4)
    #x egy 32-es torch tensor/list
    # a return value pedig egy 32*4-es torch tensor
    def forward(self, x):
        to_print = False
        if isinstance(x, torch.Tensor):
            new_x = torch.zeros([len(x), N**2])
            new_x[torch.arange(len(x)), x] = 1
            x = new_x
        else:
            x = calculate_onehot(x)
        #itt számoljuk ki a hálónk predikcióját
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        if to_print:
            print(x)
        
        return x

# itt példányosítjuk a hálót
model = DQN()
#itt az optimizer segítségével a hálónk paramétereit tudjuk automatikusan optimalizálni
optimizer = optim.Adam(model.parameters(), lr=LR)
#ezt használjuk az elvárt és a prediktált értékek különbségének kiszámítására
loss_fn = nn.HuberLoss()


#plotting the Q network
def visualize_model():
    model_input = torch.arange(N**2)
    model_output = model(model_input).detach().numpy()
    img = visualize.draw_q_table(model_output,N)
    cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imshow('image', img)
    cv2.waitKey(1)
    return img

# import ipdb; ipdb.set_trace()
# make_grid_input()

env = gym.make(
        'FrozenLake-v1',
        is_slippery = False,
        render_mode = "human",
        map_name = f"{N}x{N}"
)

def argmax(x):
    maxidxs = []
    maxval = max(x)
    for i in range(len(x)):
        if x[i] == maxval:
            maxidxs.append(i)
    return maxidxs

#a state-re ténylegesen kiválaszt egy action-t
def greedy_policy(state):
    action = random.choice(argmax(model(state)))
    return action

#ugyanaz, mint a greedy_policy csak epsilon valószínűséggel egy véletlenszerű döntést hozunk meg
def epsilon_greedy_policy(state, epsilon):
    k = torch.rand(1)[0]
    if k > epsilon:
        model_output = model(state)
        # import ipdb; ipdb.set_trace()
        action = torch.multinomial(torch.softmax(model_output,dim=0),1)
        action = action.item()
    else:
        action = random.choice([0,1,2,3])
    return action
        
#a Q-learning szabályai szerint kiszámoljuk, hogy a jelenlegi háló 
# Q-predikcióit hogyan frissítsük az új tapasztalatok alapján
def update_model():
    loss = 0
    count = 0
    for env_next_rew in balanced_memory.keys():
        for batch_idx in range(BATCH_SIZE):
            state, action, next_state, next_rew = random.choice(balanced_memory[env_next_rew])

            if next_rew == 0:
                expected_value = next_rew + DISCOUNT * max(model(next_state))
            else:
                expected_value = torch.tensor(next_rew).float()
            model_output = model(state)
            real_value = model_output[action] 
            
            loss += loss_fn(expected_value, real_value)
            count +=1
    
    loss = loss/count

    visualize_model()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    wandb.log({"loss":loss})

def get_current_epsilon(episode_index):
    current_epsilon = ((EPSILON_FINAL-EPSILON_INIT)/NUM_OF_EPISODES)*episode_index+EPSILON_INIT
    return current_epsilon

state, info = env.reset()
last_5_episodes = []
#visualize_model()
#NUM_OF_EPISODES = hányszor játszódjon végig az a folyamat, 
# hogy a start pozíciótól újrakezdi az agent a mozgást
terminated, truncated = False, False
balanced_memory = defaultdict(list)
for episode_index in range(NUM_OF_EPISODES):
    print(episode_index)
    num_steps = 0
    while True:
        num_steps += 1
        current_epsilon = get_current_epsilon(episode_index)
        action = epsilon_greedy_policy(state, current_epsilon)
        next_state, env_next_rew, terminated, truncated, info = env.step(action)

        if terminated and env_next_rew != 1:
            env_next_rew = -1
        next_rew = env_next_rew
        next_rew -= (num_steps/MAX_NUM_STEPS)

        balanced_memory[env_next_rew].append((state, action, next_state, next_rew))
        # update_model(state,next_state, next_rew, action, i)
        update_model()
        if next_rew == 1:
            dqn_state_vis = wandb.Image(
                visualize_model(),
                caption = "DQN state"
            ) 
            wandb.log({"succesful runs":dqn_state_vis})
        
        state=next_state
        wandb.log({"reward":next_rew})
        exp_counts = dict()
        for rew_type, exp_list in balanced_memory.items():
            exp_counts[rew_type] = len(exp_list)
            wandb.log({f"rew_type={rew_type}":len(exp_list)})
        if terminated or truncated:
            print(exp_counts)
            #akkor kerülünk ide, ha egy episode véget ér (roll out)
            state, info = env.reset()
            wandb.log({"episode_indexation":episode_index})
            break
        wandb.log({"current_epsilon": current_epsilon})
    #print(current_epsilon, episode_index)

#innentől kezdve a Q-learning policy-t már nem tanítjuk csak teszteljük
env = gym.make(
        'FrozenLake-v1',
        is_slippery = False,
        render_mode = "human",
        map_name = f"{N}x{N}"
)
state, info = env.reset()
for i in range(1000):
    action = epsilon_greedy_policy(state,0.1)
    state, rew, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        state, info = env.reset()