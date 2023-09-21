#%%
#Importing libraries
import numpy as np
import random
import pandas as pd
from itertools import product
#%%
#getting combinations in a integer list
q = []
for i in range(2):
    for j in range(2):
        for k in range(2):
            a = [i,j,k]
            q.append(a)
#%%
#Obtaining combinations of 0 and 1
combinations = product('01', repeat=3)
# Convert the tuples to strings
valid_combinations = [''.join(comb) for comb in combinations]
print(valid_combinations)
#%%
#Creating a datframe for counts
states = pd.DataFrame(index=valid_combinations)

#%%
#Defining functions for asynchronous 
def A(x_all, n):
    x_all[n-1][0] = int((x_all[n-2][0]) and (not x_all[n-2][1]) and (not x_all[n-2][2]))

def B(x_all, n):
    x_all[n-1][1] = int((x_all[n-2][1]) and (not x_all[n-2][0]) and (not x_all[n-2][2]))

def C(x_all, n):
    x_all[n-1][2] = int((x_all[n-2][2]) and (not x_all[n-2][0]) and (not x_all[n-2][1]))

func = [A, B, C]
#%%
#Defining function to obtain a new state
def run(x):
    x_all = []
    # x = [1,0,0]
    np.array(x_all.append(list(x)))
    x_all.append(list(x))
    f = random.choice(func)
    f(x_all, len(x_all))
    return x_all[(len(x_all)-1)]

#%%
#Running simulations

for k in range(len(q)):
    current_state = q[k]
    # Setting counts 0 for new iteration
    states['Counts'] = 0 
    state_path = []
    # #taking input
    # current_state = [0,0,0] #initial states
    # for i in range(3):
    #     current_state[i] = int(input('Enter State: \n'))
    current_state_copy = current_state

    for i in range(100):
        state_path.append(list(current_state))
        result = ''.join(map(str, current_state))
        states.loc[result, 'Counts'] += 1
        current_state = current_state_copy
        for j in range(99): 
            next_state = run(current_state) 
            result = ''.join(map(str, next_state))
            states.loc[result, 'Counts'] += 1
            current_state = list(next_state)
            state_path.append(list(current_state))

    print(current_state_copy)
    print(states)

# %%
