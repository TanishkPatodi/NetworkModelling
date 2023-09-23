#%%
# Importing libraries
import numpy as np
from itertools import product
import pandas as pd
from tqdm import tqdm
#%%
# Function to find the next state
def next_state_function(current_state_matrix, edge_matrix):
        current_state_matrix_new = np.dot(edge_matrix, current_state_matrix)
        current_state_matrix_new_2 = [
                [-1 if element[0] < -1 else 
                1 if element[0] > 1 else 
                current_state_matrix[i][0] if element[0] == 0 
                else element[0]] 
                for i, element in enumerate(current_state_matrix_new)
                ]
        return current_state_matrix_new_2
 
# Function to embed a 2x2 matrix
def embed_2x2(matrix, sub_matrix, row, col):
        matrix[row:row+2, col:col+2] = sub_matrix
 
#%%
# Simulating for different density sizes
for density in range(5,7):
    # Declaring List for matrices and list of pure states
    list_of_matrices = []
    list_of_percent_pure_states = []
    for q in tqdm(range(100)):
 
        # Defining the sparse matrix
        matrix = np.zeros((6, 6), dtype=int)
        for i in range(6):
                indices = np.random.choice(6, density, replace=False)  # Adjust range here (1-4)
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
        int_combinations_tuple = list(product([-1, 1], repeat=6))
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
            current_state_copy = current_state
            for i in range(100):
                #  state_path.append(list(current_state))
                result = ','.join(map(str, current_state))
                states.loc[result, str_combinations[k]] += 1
                current_state = current_state_copy
                for j in range(99): 
                    next_state = next_state_function(current_state_matrix, matrix) 
                    result = ','.join(str(element[0]) for element in next_state)
                    states.loc[result, str_combinations[k]] += 1
                    current_state = [element[0] for element in next_state]
                    current_state_matrix = next_state
                #      state_path.append(list(current_state))
                # print(current_state_copy)=
        # Extracting pure states
        mask = states.index.str.endswith('-1,1') | states.index.str.endswith('1,-1')
        result = states[mask]
        # result.to_csv('./Subset.csv')
        filtered = result.values.sum()
        percent_pure = filtered/640000
        list_of_percent_pure_states.append(percent_pure)
        # print(result)
        # print(list_of_percent_pure_states)
    # Saving list of matrices and percent pure states
    list_of_percent_pure_states = np.array(list_of_percent_pure_states)
    np.savetxt(f'list_of_density_{density}.csv', list_of_percent_pure_states, delimiter=',')
    list_of_matrices = np.array(list_of_matrices)
    np.save(f'list_of_matrices_D{density}.csv', list_of_matrices)
# %%