'''
This code is for BoolCheck, the network topologies Kishore bhaiya sent.
Just to verify the results.

Working correctly.

'''
#%%
import numpy as np
import random
import copy
import pandas as pd
import time
from tqdm import tqdm

#%%
def increment_or_create_key(my_dict, key):
    if key in my_dict:
        my_dict[key] += 1
    else:
        my_dict[key] = 1
#%%


# Define the desired order of nodes
desired_order = ['CDH1', 'FOXC2',
'GSC',
'KLF8',
'SNAI1',
'SNAI2',
'TCF3',
'TGFbeta',
'TWIST1',
'TWIST2',
'VIM',
'ZEB1',
'ZEB2',
'miR101',
'miR141',
'miR200a',
'miR200b',
'miR200c',
'miR205',
'miR30c',
'miR34a',
'miR9'
]  # Replace with your specific node names

# Load your data from the file
with open('../../../Downloads/BoolCheck/EMT_RACIPE.topo', 'r') as f:
    lines = f.readlines()

# Get column names
column_names = lines[0].strip().split()

# Create a list of unique nodes in the desired order
nodes = desired_order

# Create an empty edge matrix filled with zeros
num_nodes = len(nodes)
edge_matrix = np.zeros((num_nodes, num_nodes))

# Populate the edge matrix
for line in lines[1:]:
    source, target, interaction = line.strip().split()
    source_idx = nodes.index(source)
    target_idx = nodes.index(target)
    
    if interaction == '1':  # Activation
        edge_matrix[source_idx, target_idx] = 1
    elif interaction == '2':  # Inhibition
        edge_matrix[source_idx, target_idx] = -1

print("Column Names:", column_names)
print("Edge Matrix:")
edge_matrix = edge_matrix.T
print(np.array(edge_matrix))


#%%
edge_matrix = np.array([[0,0,-1,0],
          [0,0,1,-1],
          [-1,0,1,-1],
          [0,0,-1,0]]).T
#%%
edge_matrix=np.array([[0,1, 0],
                      [-1,0,0],
                      [-1,0,0]]).T

# s = time.time()
# combination = np.random.choice([-1,1], size=(len(matrix),1))
# print(combination)
# node = np.random.randint(len(matrix)-1)
# ns = copy.deepcopy(combination)
# ns[node] = np.matmul(matrix[node], combination)
# ns = np.sign(ns)
# ns[ns == 0] = combination[ns == 0]
# e = time.time()
# print(combination)
# print(ns)
# print(e-s)
# %%
states = {}
for i in tqdm(range(5000)):
    # combination = np.random.choice([-1,1], size=(len(edge_matrix),1))
    combination = [random.choice([-1, 1]) for _ in range(len(edge_matrix))]
    x = 0
    limit = 5000
    while x < 300 and limit > 0:
        limit -= 1
        node = random.randint(0, len(edge_matrix)-1)
        ns = copy.deepcopy(combination)
        ns[node] = np.matmul(edge_matrix[node], combination)
        ns = np.sign(ns)
        if ns[node] == 0:
            ns[node] = combination[node]
        # ns[ns == 0] = combination[ns == 0]
        if np.array_equal(ns, combination):
            x += 1
        else:
            x = 0
        combination = copy.deepcopy(ns)
    if x == 300:       
        result = ','.join(map(str, ns))
        # result = ','.join(str(element[0]) for element in ns)
        increment_or_create_key(states, result)


states_df = pd.DataFrame(states.items(), columns=['States', 'Counts']).set_index('States')
total = states_df.values.sum()
mask = states_df.index.str.endswith('-1,1') | states_df.index.str.endswith('1,-1')
result = states_df[mask]
filtered = result.values.sum()
states_df.index = states_df.index.str.replace('-1','0')
states_df['Fraction'] = states_df['Counts'].apply(lambda x: x/total)
print(states_df)
if total > 0:
    percent_pure = filtered/total
    print(percent_pure)
#%%

states_df.to_csv('RACIPE_kishore.csv')

# %%
