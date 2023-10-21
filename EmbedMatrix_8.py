'''
I aim to efficiently multiprocess the code and reduce the multiple for loops in this code.

Completed: took all parameter set in one, which solves our problem - most cores wait for others to get completed, just to start for next cycle.
Inshort - Earlier multiprocessing was i/o based, changed that by doing all at once
'''
#%%
# from tqdm import tqdm
import copy
import numpy as np
import pandas as pd
import time
import multiprocessing
import random
import networkx as nx
from tqdm import tqdm

#%%
def increment_or_create_key(my_dict, key):
    if key in my_dict:
        my_dict[key] += 1
    else:
        my_dict[key] = 1

# Function to embed a 2x2 matrix
def embed_2x2(matrix, sub_matrix, row, col):
        matrix[row:row+2, col:col+2] = sub_matrix

def init_pool():
    np.random.seed(multiprocessing.current_process().pid)

def generate_matrix(n, d):
    matrix = np.zeros((n, n), dtype=int)
    indices = []
    while len(indices) < n*d:
        indices = np.random.choice(n*n, n*d, replace=False) 
        # Ensure no element on the diagonal gets updated #Clever Idea!
        indices = [idx for idx in indices if idx % (n+1) != 0]

    matrix.flat[indices] = np.random.choice([-1, 1], len(indices))
    return matrix
#%%
def simulate(density, node, q):
        
        # Parameters
        steadystate = 300
        initial_conditions = 1000
        breakif = 300

        edge_matrix = np.array(matt[q])

        states = {}
        for i in range(initial_conditions):
            # combination = np.random.choice([-1,1], size=(len(edge_matrix),1))
            combination = [random.choice([-1, 1]) for _ in range(len(edge_matrix))]
            x = 0
            limit = 5000
            while x < steadystate and limit > 0:
                limit -= 1
                rnode = random.randint(0, len(edge_matrix)-1)
                ns = copy.deepcopy(combination)
                ns[rnode] = np.matmul(edge_matrix[rnode], combination)
                ns = np.sign(ns)
                if ns[rnode] == 0:
                    ns[rnode] = combination[rnode]
                # ns[ns == 0] = combination[ns == 0]
                if np.array_equal(ns, combination):
                    x += 1
                else:
                    x = 0
                combination = copy.deepcopy(ns)
            if x == steadystate:       
                result = ','.join(map(str, ns))
                # result = ','.join(str(element[0]) for element in ns)
                increment_or_create_key(states, result)
            if i == breakif and len(states)==0:
                break

        states_df = pd.DataFrame(states.items(), columns=['States', 'Counts']).set_index('States')
        total = states_df.values.sum()
        mask = states_df.index.str.endswith(',-1.0,1.0') | states_df.index.str.endswith(',1.0,-1.0')
        result = states_df[mask]
        filtered = result.values.sum()
        states_df.index = states_df.index.str.replace('-1','0')
        states_df['Fraction'] = states_df['Counts'].apply(lambda x: x/total)
        print(states_df)
        # states_df.to_clipboard()
        if total > 0:
            percent_pure = filtered/total
            dicti_all[f'{node}N-{density}D-R{q}'] = percent_pure
            # return percent_pure
            # print(percent_pure)

node = 7
density = 2
matt = np.load(f'../100 Random Matrices/{node}N-{density}D.npy')

#%%
if __name__ == '__main__':
    start = time.time()
    dicti_all = {}
    pool = multiprocessing.Pool(processes=1, initializer=init_pool)
    parameter_all = []
    for node in range(7,23,5):
        for density in range(2,7,2):
            p = [(density, node)]*100
            pn = [(density, node, i) for i, (density, node) in enumerate(p)]
            parameter_all += pn
    pool.starmap(simulate, parameter_all)
    pool.close()
    pool.join()
    result_df = pd.DataFrame(dicti_all.items(), columns=['Combination', 'Percent Pure'])
    result_df.to_csv('./ResultAll_OptiMult.csv', sep=',')
    end = time.time()
    print(f"Time Elapsed:{end - start} ")
#%%


# %%
