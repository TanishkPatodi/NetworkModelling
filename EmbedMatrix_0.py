'''
Creates a list and checks if last n elemets are same
This code was first succesful code, its redundant but does works. Several changes are made after this
'''

#%%
# Importing libraries
import numpy as np
from itertools import product, chain
import pandas as pd
from tqdm import tqdm
import random
import time
#%%
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
    if len(lst) < 70:
        return False
    
    last_10_elements = lst[-70:]
    return all(element == last_10_elements[0] for element in last_10_elements)
# Function to embed a 2x2 matrix
def embed_2x2(matrix, sub_matrix, row, col):
        matrix[row:row+2, col:col+2] = sub_matrix
 
#%%
start = time.time()
# Simulating for different density sizes
for density in range(2,4,2):
        # Declaring List for matrices and list of pure states
        list_of_matrices = []
        list_of_percent_pure_states = []
        for q in tqdm(range(100)):
                # Defining the sparse matrix

                # matrix = [[ 0, -1,  0,  0, -1,  0, -1],        [-1,  0,  0, -1,  0,  0,  1],        [ 1,  0,  0,  0,  1, -1,  0],        [ 0,  1, -1,  0,  0,  0,  0],        [ 0,  0,  1,  1,  0,  1,  0],        [ 0,  1,  0,  1,  0,  0, -1],        [-1,  0,  0,  0, -1, -1,  0]]
                matrix = np.zeros((6, 6), dtype=int)
                for i in range(6):
                        indices = np.random.choice(6, density, replace=False) 
                        matrix[i, indices] = np.random.choice([-1, 1], density)

                        if 1 not in matrix[i]:
                                matrix[i, np.random.choice(6)] = 1
                        if -1 not in matrix[i]:
                                matrix[i, np.random.choice(6)] = -1

                # Defining the edge matrix
                embed_matrix = np.full(shape=(2,2), fill_value= -1)
                np.fill_diagonal(embed_matrix, 1) #change diagonal value to 1/0 for SA


                # Embedding matrix
                embed_2x2(matrix, embed_matrix, len(matrix)-2, len(matrix[1])-2)
                list_of_matrices.append(matrix)


                # Running simulations
                # Obtaining combinations of 0 and 1 in integer format
                int_combinations_tuple = list(product([-1, 1], repeat=7))
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
                        for i in range(100):
                                state_path = []
                                result = ','.join(map(str, current_state))
                                state_path.append(list(current_state))
                        #   states.loc[result, str_combinations[k]] += 1
                                current_state_matrix = current_state_matrix_copy
                                current_state = current_state_copy
                                for j in range(99): 
                                        next_state = next_state_function_Async(current_state_matrix, matrix) 
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
                # Extracting pure states
                mask = states.index.str.endswith('-1,1') | states.index.str.endswith('1,-1')
                result = states[mask]
                # result.to_csv('./Subset.csv')
                filtered = result.values.sum()
                percent_pure = filtered/6400
                print(percent_pure)
                list_of_percent_pure_states.append(percent_pure)
                # print(result)
                # print(list_of_percent_pure_states)
        # Saving list of matrices and percent pure states
        list_of_percent_pure_states = np.array(list_of_percent_pure_states)
        np.savetxt(f'list_of_density_{density}.csv', list_of_percent_pure_states, delimiter=',')
        # list_of_matrices = np.array(list_of_matrices)
        # np.save(f'list_of_matrices_D{density}.csv', list_of_matrices)
end = time.time()
print(f"Time Elapsed:{end - start} ")
# %%
