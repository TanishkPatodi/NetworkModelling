'''
This code is just a reattempt for the code...
It didn't work.

Problem: 1> The definition of edge matrix is different and wrong.
2> Ising formalism was misunderstood...

'''

#%%
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
    if len(lst) < 10:
        return False
    
    last_10_elements = lst[-10:]
    return all(element == last_10_elements[0] for element in last_10_elements)
# Function to embed a 2x2 matrix
def embed_2x2(matrix, sub_matrix, row, col):
        matrix[row:row+2, col:col+2] = sub_matrix
 
#%%
matrix = [[ 0,  0,  0, -1, -1,  0, -1],        [ 1,  0, -1,  0,  0,  1,  0],        [ 1,  1,  0,  0,  0,  0,  0],        [ 0,  1,  0,  0, -1, -1,  0],        [ 0,  0,  1, -1,  0,  0, -1],        [ 0,  0,  1,  0,  1,  0, -1],        [ 0, -1,  0,  1,  0, -1,  0]]
node = 7
num_combinations = 100
int_comb_sample = []
for i in range(num_combinations):
    combination = np.random.choice([-1,1], size=(node,1))
    int_comb_sample.append(combination)

for initial_condition in int_comb_sample:
     all_counts = 0
     pure_count = 0
     initial_condition_copy = list(initial_condition)
     for i in range(100):
          initial_condition = initial_condition_copy
          for j in range(100):
                updated_condition = np.matmul(matrix, initial_condition)
                if np.array_equal(updated_condition, initial_condition):
                    all_counts += 1
                    if (updated_condition[-1][0] == 1 and updated_condition[-2][0] == -1) or (updated_condition[-1][0] == -1 and updated_condition[-2][0] == 1):
                         pure_count += 1
                    break
                else:
                    updated_condition = [
                                        [-1 if element[0] < -1 else 
                                        1 if element[0] > 1 else 
                                        initial_condition[i][0] if element[0] == 0 
                                        else element[0]] 
                                        for i, element in enumerate(updated_condition)]
                    initial_condition = list(updated_condition)
     print(pure_count/all_counts)       
                
                         
     

# %%
