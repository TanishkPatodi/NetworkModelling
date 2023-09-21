#%%
#Importing libraries
import numpy as np
import random
import pandas as pd
from itertools import product
#%%
#Defining function to obtain a new state
def run(x):
    x_all = []
    np.array(x_all.append(list(x)))
    x_all.append(list(x))
    x_all.append(list(x))
    to_return = n_node(x_all, len(x_all))
    return to_return
#%%
def n_node(list_of_nodes, length):
    node = random.randint(0, len(list_of_nodes[0])-1)
    # Apply NOT operation to every element except the one at except_index
    list_of_nodes[1] = [not element if i != node 
                        else element 
                        for i, element in enumerate(list_of_nodes[0])]
    list_of_nodes[1] = list_of_nodes[1] # For removing self activation
    # Apply AND operation between each element and the rest of the elements
    list_of_nodes[2][node] = any(list_of_nodes[1]) #Change any to all()/any() for async AND/OR
    list_of_nodes[2] = [int(element) for element in list_of_nodes[2]]
    return list_of_nodes[2]

#Running simulations 
#taking the input for number of nodes in the system
n = int(input('Enter the number of nodes in a system'))

#Obtaining combinations of 0 and 1 for counts
combinations = product('01', repeat=n)
# Convert the tuples to strings
valid_combinations = [','.join(comb) for comb in combinations]

#Obtaining combinations of 0 and 1 in integer format
int_combinations = list(product([0, 1], repeat=n))

#Creating a datframe for counts
states = pd.DataFrame(index=valid_combinations)

for k in range(len(int_combinations)):
    current_state = int_combinations[k]
    # Setting counts 0 for new iteration
    states[valid_combinations[k]] = 0 
    state_path = []
    current_state_copy = current_state

    # f = random.choice(func)  #check

    for i in range(100):
        state_path.append(list(current_state))
        result = ','.join(map(str, current_state))
        states.loc[result, valid_combinations[k]] += 1
        current_state = current_state_copy
        for j in range(99): 
            next_state = run(current_state) 
            result = ','.join(map(str, next_state))
            states.loc[result, valid_combinations[k]] += 1
            current_state = list(next_state)
            state_path.append(list(current_state))

    # print(current_state_copy)
print(states)
# %%
states.to_csv('./AsyncAND_Triad_SA.csv')
# %%
