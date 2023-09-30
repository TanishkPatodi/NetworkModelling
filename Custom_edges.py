#%%
import numpy as np
from itertools import product, chain
import pandas as pd
import random
# %%
# Defining functions
random.seed(42)
def next_state_function_Async(current_state_matrix, edge_matrix):
    temp = list(current_state_matrix)
    current_state_matrix_new = np.dot(edge_matrix, current_state_matrix)
    current_state_matrix_new_2 = [
            [-1 if element[0] < -1 else 
            1 if element[0] > 1 else 
            current_state_matrix[i][0] if element[0] == 0 
            else element[0]] 
            for i, element in enumerate(current_state_matrix_new)
            ]
    node = random.randint(0, len(current_state_matrix)-1)
    temp[node] = current_state_matrix_new_2[node] 
      
    return temp 

def next_state_function_sync(current_state_matrix, edge_matrix):
       current_state_matrix_new = np.dot(edge_matrix, current_state_matrix)
       current_state_matrix_new_2 = [
              [-1 if element[0] < -1 else 
              1 if element[0] > 1 else 
              current_state_matrix[i][0] if element[0] == 0 
              else element[0]] 
              for i, element in enumerate(current_state_matrix_new)
              ]
       return current_state_matrix_new_2 #For sync update

def are_last_10_elements_same(lst):
    if len(lst) < 30:
        return False
    
    last_10_elements = lst[-30:]
    return all(element == last_10_elements[0] for element in last_10_elements)

#%%
#Running simulations
#taking the input for number of nodes in the system
n = int(input('Enter the number of nodes in a system'))

# Defining the edge matrix
edge_matrix =  [[1, 1, -1, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, -1, 1, 1]]
#Obtaining combinations of 0 and 1 in integer format
int_combinations_tuple = list(product([-1, 1], repeat=n))

# Convert the tuples to lists
int_combinations = [list(combination) for combination in int_combinations_tuple]
str_combinations = [','.join(map(str, combination)) for combination in int_combinations_tuple]

#Creating a datframe for counts
states = pd.DataFrame(index=str_combinations)
for k in range(len(int_combinations)):
    current_state = int_combinations[k]
    #taking current state as matrix
    current_state_matrix = []
    for element in int_combinations[k]:
       current_state_matrix.append([element])
    # Creating new column for new iteration/state
    states[str_combinations[k]] = 0 
    current_state_matrix_copy = list(current_state_matrix)
    current_state_copy = current_state
    for i in range(1000):
        state_path = []
        result = ','.join(map(str, current_state))
        state_path.append(list(current_state))
      #   states.loc[result, str_combinations[k]] += 1
        current_state_matrix = current_state_matrix_copy
        current_state = current_state_copy
        for j in range(99): 
            next_state = next_state_function_Async(current_state_matrix, edge_matrix) 
            current_state = list(chain.from_iterable(next_state))
            result = ','.join(str(element[0]) for element in next_state)
            # states.loc[result, str_combinations[k]] += 1
            current_state = [element[0] for element in next_state]
            state_path.append(list(current_state))
            # print(state_path, "\n\n\n\n")
            current_state_matrix = next_state
            if are_last_10_elements_same(state_path):
                states.loc[result, str_combinations[k]] += 1
                break
            else:
                continue
            
            

       #      state_path.append(list(current_state))
    # print(current_state_copy)
print(states)
# %%
states.to_csv('./D6_Async.csv')
# %%
