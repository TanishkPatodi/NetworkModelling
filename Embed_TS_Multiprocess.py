#%%
import numpy as np
from itertools import product, chain
import pandas as pd
from tqdm import tqdm
import random
import multiprocessing
import time
#%%
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

        #     if 1 not in matrix[i]:
        #             matrix[i, np.random.choice(6)] = 1
        #     if -1 not in matrix[i]:
        #             matrix[i, np.random.choice(6)] = -1

    # Defining the edge matrix
    embed_matrix = np.full(shape=(2,2), fill_value= -1)
    np.fill_diagonal(embed_matrix, 1) #change diagonal value to 1/0 for SA


    # Embedding matrix
    embed_2x2(matrix, embed_matrix, len(matrix)-2, len(matrix[1])-2)
#     list_of_matrices.append(matrix)


    # Running simulations
    # Obtaining combinations of 0 and 1 in integer format
    int_combinations_tuple = list(product([-1, 1], repeat=node))
    # Convert the tuples to lists
    int_combinations = [list(combination) for combination in int_combinations_tuple]
    int_comb_sample = random.sample(int_combinations, len(int_combinations))
#     str_combinations = [','.join(map(str, combination)) for combination in int_combinations_tuple]
    #Creating a datframe for counts
    states = {}
    for k in range(len(int_comb_sample)):
            current_state = int_comb_sample[k]
            #taking current state as matrix
            current_state_matrix = []
            for element in int_comb_sample[k]:
                    current_state_matrix.append([element])
            current_state_matrix_copy = list(current_state_matrix)
            current_state_copy = current_state
            for i in range(100):
                    state_path = []
                    state_path.append(list(current_state))
                    current_state_matrix = current_state_matrix_copy
                    current_state = current_state_copy
                    for j in range(100):
                            next_state = next_state_function_Async(current_state_matrix, matrix) 
                            current_state = list(chain.from_iterable(next_state))
                            result = ','.join(str(element[0]) for element in next_state)
                            # states.loc[result, str_combinations[k]] += 1
                            current_state = [element[0] for element in next_state]
                            state_path.append(list(current_state))
                            # print(state_path, "\n\n\n\n")
                            current_state_matrix = next_state
                            if are_last_10_elements_same(state_path):
                                increment_or_create_key(states, result)
                                break
                            else:
                                continue

    states_df = pd.DataFrame(states.items(), columns=['States', 'Counts']).set_index('States')
#     print(states_df)
    total = states_df['Counts'].sum()
#     print(f'Total:{total}')
    # Extracting pure states
    mask = states_df.index.str.endswith('-1,1') | states_df.index.str.endswith('1,-1')
    
    result = states_df[mask]
    # print(result)
    # # result.to_csv('./Subset.csv')
    
    filtered = result['Counts'].sum()
#     print(filtered)
    percent_pure = filtered/total
    return percent_pure

def init_pool():
        np.random.seed(multiprocessing.current_process().pid)

#%%
if __name__ == '__main__':
        
        for node in range(6,7):
            start = time.time()
            for density in range(2,3):    
                    # list_of_matrices = []
                    list_of_percent_pure_states = []
                    pool = multiprocessing.Pool(processes=8, initializer=init_pool)
                    parameter = [(density, node)]*10
                    list_of_percent_pure_states = pool.starmap(simulate, parameter)
                    pool.close()
                    pool.join()
                    print(list_of_percent_pure_states)
                    # Saving list of matrices and percent pure states
                    list_of_percent_pure_states = np.array(list_of_percent_pure_states)
                    np.savetxt(f'{node}N_{density}d_sub_all_100each.csv', list_of_percent_pure_states, delimiter=',')
                    # list_of_matrices = np.array(list_of_matrices)
                    # np.save(f'list_of_matrices_D{density}.csv', list_of_matrices)
            end = time.time()
            print(f"Time Elapsed:{end - start} ")
        
# %%
