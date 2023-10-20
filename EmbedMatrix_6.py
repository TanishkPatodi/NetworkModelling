'''
This code is after removing checks to find every posible path for a given initial condition
SS = if a state repeats for n number of times

Works!
Results match!
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
def simulate(density, node):
        
        # Parameters
        steadystate = 300
        initial_conditions = 1000
        breakif = 300
        

        # Create a directed graph from the adjacency matrix
        edge_matrix = generate_matrix(node, density)
        embed_matrix = np.full(shape=(2,2), fill_value= -1)
        np.fill_diagonal(embed_matrix, 0) #change diagonal value to 1/0 for SA
        embed_2x2(edge_matrix, embed_matrix, len(edge_matrix)-2, len(edge_matrix[1])-2)  
        G = nx.DiGraph(edge_matrix)      
        while nx.is_connected(G.to_undirected()) == False:
            edge_matrix = generate_matrix(node, density)
            embed_matrix = np.full(shape=(2,2), fill_value= -1)
            np.fill_diagonal(embed_matrix, 0) #change diagonal value to 1/0 for SA
            embed_2x2(edge_matrix, embed_matrix, len(edge_matrix)-2, len(edge_matrix[1])-2)
            G = nx.DiGraph(edge_matrix)
        print(edge_matrix)

        '''
        # To pass custom edge network
        edge_matrix=np.array([[0, 0, 0, 1, 0, 0, 1, -1, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1],
                                [0, -1, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0],
                                [0, 1, 0, 0, -1, 0, -1, 1, 0, 0, 0, 0],
                                [-1, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 1],
                                [0, 0, 0, -1, 0, 0, 0, 0, 1, -1, -1, 0],
                                [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, -1],
                                [0, -1, 0, -1, 0, 0, 0, 0, 0, 1, -1, 0],
                                [-1, 0, 0, 0, 0, 1, 0, 0, 0, -1, 1, 0],
                                [0, 0, 1, 0, 0, 1, 0, -1, 1, 0, 0, 0],
                                [0, 0, 0, 1, 1, -1, 0, 0, 1, 0, 0, -1],
                                [-1, 0, 1, 0, 0, -1, 1, 0, 0, 0, -1, 0]
                                ])
        '''


        states = {}
        for i in range(initial_conditions):
            # combination = np.random.choice([-1,1], size=(len(edge_matrix),1))
            combination = [random.choice([-1, 1]) for _ in range(len(edge_matrix))]
            x = 0
            limit = 1000
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
        mask = states_df.index.str.endswith(',-1,1') | states_df.index.str.endswith(',1,-1')
        result = states_df[mask]
        filtered = result.values.sum()
        states_df.index = states_df.index.str.replace('-1','0')
        states_df['Fraction'] = states_df['Counts'].apply(lambda x: x/total)
        print(states_df)
        states_df.to_clipboard()
        if total > 0:
            percent_pure = filtered/total
            return percent_pure
            # print(percent_pure)


#%%
if __name__ == '__main__':
    for node in range(7,23,5):
        for density in tqdm(range(2,7,2)):
            start = time.time()
            list_of_percent_pure_states = []
            pool = multiprocessing.Pool(processes=100, initializer=init_pool)
            parameter = [(density, node)]*100
            list_of_percent_pure_states = pool.starmap(simulate, parameter)
            pool.close()
            pool.join()
            print(list_of_percent_pure_states)
            list_of_percent_pure_states = np.array(list_of_percent_pure_states)
            filtered = list_of_percent_pure_states[~np.isnan(list_of_percent_pure_states)]
            #np.savetxt(f'./Results16O{node}N_{density}d.csv', list_of_percent_pure_states, delimiter=',')
            end = time.time()
            print(f"Time Elapsed:{end - start} ")
            
# %%
