#%%
import numpy as np
import pandas as pd
import random
import multiprocessing
import time
#%%
def next_state_function_Async(current_state_matrix, edge_matrix):
    temp = list(current_state_matrix)
    current_state_matrix_new = np.dot(edge_matrix, current_state_matrix)
    current_state_matrix_new_2 = [[current_state_matrix[i][0] if element[0] == 0 
                                  else element[0]] 
                                  for i, element in enumerate(current_state_matrix_new)]
    current_state_matrix_new = np.sign(current_state_matrix_new)
    node = random.randint(0, len(current_state_matrix)-1)
    temp[node] = current_state_matrix_new_2[node] 
        
    return temp 
 
def next_state_function_sync(current_state_matrix, edge_matrix):
       current_state_matrix_new = np.dot(edge_matrix, current_state_matrix)
       current_state_matrix_new = np.sign(current_state_matrix_new)
       current_state_matrix_new_2 = [[current_state_matrix[i][0] if element[0] == 0 
                                  else element[0]] 
                                  for i, element in enumerate(current_state_matrix_new)]
       return current_state_matrix_new_2 #For sync update
 
def are_last_10_elements_same(lst):
    if len(lst) < 70:
        return False
 
    last_10_elements = lst[-70:]
    return all(element == last_10_elements[0] for element in last_10_elements)
 
# Function to embed a 2x2 matrix
def embed_2x2(matrix, sub_matrix, row, col):
        matrix[row:row+2, col:col+2] = sub_matrix
 
def increment_or_create_key(my_dict, key):
    if key in my_dict:
        my_dict[key] += 1
    else:
        my_dict[key] = 1
 
 
def simulate(density, node):
    # Defining the sparse matrix
    matrix = np.zeros((node, node), dtype=int)
    for i in range(node):
            indices = np.random.choice(node, density, replace=False) 
            matrix[i, indices] = np.random.choice([-1, 1], density)
 
    # Defining the edge matrix
    embed_matrix = np.full(shape=(2,2), fill_value= -1)
    np.fill_diagonal(embed_matrix, 0) #change diagonal value to 1/0 for SA
    embed_2x2(matrix, embed_matrix, len(matrix)-2, len(matrix[1])-2)
    num_combinations = 50000
    
    int_comb_sample = []
    while len(int_comb_sample) < num_combinations:
        combination = [random.choice([-1, 1]) for _ in range(node)]
        if combination not in int_comb_sample:
                int_comb_sample.append(combination)
    states = {}
    for k in range(len(int_comb_sample)):
            current_state_matrix = []
            for element in int_comb_sample[k]:
                    current_state_matrix.append([element])
            current_state_matrix_copy = list(current_state_matrix)
            for i in range(100):
                    current_state_matrix = current_state_matrix_copy
                    for j in range(100):
                            next_state = next_state_function_Async(current_state_matrix, matrix) 
                            all_same = all(elem1 == elem2 for elem1, elem2 in zip(current_state_matrix, next_state))
                            if all_same:
                                result = ','.join(str(element[0]) for element in next_state)
                                increment_or_create_key(states, result)
                                break
                            else:
                                current_state_matrix = next_state

 
    states_df = pd.DataFrame(states.items(), columns=['States', 'Counts']).set_index('States')
    #print(states_df)
    total = states_df.values.sum()
    mask = states_df.index.str.endswith('-1,1') | states_df.index.str.endswith('1,-1')
 
    result = states_df[mask]
    filtered = result.values.sum()
    if total > 0:
        percent_pure = filtered/total
        return percent_pure
 
#%%
if __name__ == '__main__':
    for node in range(17,23,5):
        for density in range(2,7,2):
            start = time.time()
            list_of_percent_pure_states = []
            pool = multiprocessing.Pool(processes=100)
            parameter = [(density, node)]*100
            list_of_percent_pure_states = pool.starmap(simulate, parameter)
            pool.close()
            pool.join()
            print(list_of_percent_pure_states)
            list_of_percent_pure_states = np.array(list_of_percent_pure_states)
            np.savetxt(f'{node}N_{density}d_Sub64_100_each_state.csv', list_of_percent_pure_states, delimiter=',')
            end = time.time()
            print(f"Time Elapsed:{end - start} ")
# %%
