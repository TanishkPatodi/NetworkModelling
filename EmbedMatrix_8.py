'''
I aim to efficiently multiprocess the code and reduce the multiple for loops in this code.

Completed: took all parameter set in one, which solves our problem - most cores wait for others to get completed, just to start for next cycle.
Inshort - Earlier multiprocessing was i/o based, changed that by doing all at once
'''
#%%
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
        matrix[row:row+3, col:col+3] = sub_matrix

def init_pool():
    np.random.seed(multiprocessing.current_process().pid)

def create_random_matrix(n, d):
    matrix = np.zeros((n, n))

    indices = [(i, j) for i in range(n) for j in range(n) if i != j]
    np.random.shuffle(indices)

    for i in range(n*d):
        row, col = indices[i]
        value = np.random.choice([-1, 1])
        matrix[row, col] = value

    empty_rows = np.where(~matrix.any(axis=1))[0]
    empty_cols = np.where(~matrix.any(axis=0))[0]

    for row in empty_rows:
        col = np.random.choice(np.delete(np.arange(n), row))
        matrix[row, col] = np.random.choice([-1, 1])

    for col in empty_cols:
        row = np.random.choice(np.delete(np.arange(n), col))
        matrix[row, col] = np.random.choice([-1, 1])

    embed_matrix = np.full(shape=(2,2), fill_value= -1)
    np.fill_diagonal(embed_matrix, 0) #change diagonal value to 1/0 for SA
    embed_2x2(matrix, embed_matrix, len(matrix)-2, len(matrix[1])-2)

    return matrix


def switch_case(argument):
    match argument:
        case '7N2D':
            matt = np.load(f'../100 Random Matrices/7N-2D.npy')
            return matt
        case '7N4D':
            matt = np.load(f'../100 Random Matrices/7N-4D.npy')
            return matt
        case '7N6D':
            matt = np.load(f'../100 Random Matrices/7N-6D.npy')
            return matt
        case '12N2D':
            matt = np.load(f'../100 Random Matrices/12N-2D.npy')
            return matt
        case '12N4D':
            matt = np.load(f'../100 Random Matrices/12N-4D.npy')
            return matt
        case '12N6D':
            matt = np.load(f'../100 Random Matrices/12N-6D.npy')
            return matt
        case '17N2D':
            matt = np.load(f'../100 Random Matrices/17N-2D.npy')
            return matt
        case '17N4D':
            matt = np.load(f'../100 Random Matrices/17N-4D.npy')
            return matt
        case '17N6D':
            matt = np.load(f'../100 Random Matrices/17N-6D.npy')
            return matt
        case '22N2D':
            matt = np.load(f'../100 Random Matrices/22N-2D.npy')
            return matt
        case '22N4D':
            matt = np.load(f'../100 Random Matrices/22N-4D.npy')
            return matt
        case '22N6D':
            matt = np.load(f'../100 Random Matrices/22N-6D.npy')
            return matt

#%%
def simulate(density, node, dicti_all, q):
        
        # Parameters
        steadystate = 300
        initial_conditions = 10000
        breakif = 300
        
        # impliment switch if input matrices are predetermined.
        matt = switch_case(f'{node}N{density}D')
        
        edge_matrix = np.array(matt[q])
        embed_matrix = np.full(shape=(3,3), fill_value= -1)
        np.fill_diagonal(embed_matrix, 0) #change diagonal value to 1/0 for SA
        embed_2x2(edge_matrix, embed_matrix, len(edge_matrix)-3, len(edge_matrix[1])-3)


        # edge_matrix = create_random_matrix(node, density)
        states = {}
        for i in range(initial_conditions):
            # combination = np.random.choice([-1,1], size=(len(edge_matrix),1))
            combination = [random.choice([-1, 1]) for _ in range(len(edge_matrix))]
            x = 0
            limit = 10000
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
        maskF1 = states_df.index.str.endswith('-1.0,-1.0,1.0') | states_df.index.str.endswith('1.0,-1.0,-1.0') | states_df.index.str.endswith('-1.0,1.0,-1.0')
        resultF1 = states_df[maskF1]
        filteredF1 = resultF1.values.sum()
        maskF2 = states_df.index.str.endswith('-1.0,1.0,1.0') | states_df.index.str.endswith('1.0,-1.0,1.0') | states_df.index.str.endswith('1.0,1.0,-1.0')    
        resultF2 = states_df[maskF2]
        filteredF2 = resultF2.values.sum() 
        maskF3 = states_df.index.str.endswith('1.0,1.0,1.0')                  
        resultF3 = states_df[maskF3]    
        filteredF3 = resultF3.values.sum()
        # states_df.index = states_df.index.str.replace('-1','0')
        # states_df['Fraction'] = states_df['Counts'].apply(lambda x: x/total)
        # print(states_df)
        # states_df.to_clipboard()
        if total > 0:
            F1 = filteredF1/total
            F2 = filteredF2/total
            F3 = filteredF3/total
            # dicti_all.update({f'{node}N-{density}D-R{q}' : percent_pure})
            dicti_all[f'{node}N-{density}D-R{q}'] = F1, F2, F3
            # return percent_pure
            # print(percent_pure)
            # return ([{f'{node}N-{density}D-R{q}' : percent_pure}])



#%%
if __name__ == '__main__':
    start = time.time()
    manager = multiprocessing.Manager()
    dicti_all = manager.dict()
    pool = multiprocessing.Pool(processes=14, initializer=init_pool)
    parameter_all = []
    # matt = np.load(f'./Matrices/{7}N-{2}D.npy')
    for node in range(7,23,5):
       for density in range(2,7,2):
           p = [(density, node, dicti_all)]*100
           pn = [(density, node, dicti_all, i) for i, (density, node, dicti_all) in enumerate(p)]
           parameter_all += pn
    pool.starmap(simulate, parameter_all)
    pool.close()
    pool.join()
    print(dicti_all)
    data = [(key, value[0], value[1], value[2]) for key, value in dicti_all.items()]
    df = pd.DataFrame(data, columns=['States', 'F1', 'F2', 'F3'])
    #result_df = pd.DataFrame(dicti_all.items(), columns=['Combination', 'Percent Pure'])
    # print(result_df)
    df.to_csv('./ResultAll_OptiMultTriad.csv', sep=',')
    end = time.time()
    print(f"Time Elapsed:{end - start} ")
#%%