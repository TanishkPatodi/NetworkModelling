'''

For Running predefined matrices, for checking purposes.

       Same code as previous - just multiprocessing removed.
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
        initial_conditions = 10000
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
        # states_df.index = states_df.index.str.replace('-1','0')
        # states_df['Fraction'] = states_df['Counts'].apply(lambda x: x/total)
        # print(edge_matrix)
        # print(states_df)
        # states_df.to_clipboard()
        if total > 0:
            percent_pure = filtered/total
            return percent_pure
            # print(percent_pure)


for node in range(7,10,5):
    for density in range(2,4,2):
        list_of_percent_pure_states = []
        matt = np.load(f'../100 Random Matrices/{node}N-{density}D.npy')
        for w in tqdm(range(100)):
            list_of_percent_pure_states.append(simulate(node, density, w))
        print(f'{node}N-{density}D : \n {list_of_percent_pure_states}')
        start = time.time()
        
        
# %%


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a DataFrame named 'df' in long format with columns 'Combination' and 'Percent_Pure'
# For example:
# df = pd.DataFrame({'Combination': ['A', 'B', 'C', 'A', 'B', 'C'],
#                    'Percent_Pure': [30, 40, 50, 60, 70, 80]})

df = pd.read_csv('./ResultAll_OptiMult.csv', sep=',')
df = df.drop(df.columns[0], axis=1)
df['Combination'] = df['Combination'].str.split('-R').str[0]
#%%
fig, ax = plt.subplots(figsize=(10,6))

sns.boxplot(data=df, x='Combination', y='Percent Pure', showmeans = True)
plt.xlabel('Nodes-Density')
plt.xticks(rotation = 90)
# %%
# Calculate means
means = df.groupby('Category')['Value'].mean()

# Add mean lines
for i, mean in enumerate(means):
    plt.axhline(mean, color='red', linestyle='dashed', linewidth=1)
    plt.text(i, mean, f'Mean={mean:.2f}', va='center', ha='center', color='red', fontsize=10)

# Add labels and title
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Boxplot with Means')

# Show the plot
plt.show()