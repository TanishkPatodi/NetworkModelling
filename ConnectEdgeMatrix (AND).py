#%%
import numpy as np
import networkx as nx

# Function to embed a 2x2 matrix
def embed_2x2(matrix, sub_matrix, row, col):
        matrix[row:row+2, col:col+2] = sub_matrix

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

    return matrix

# Define the size of the matrix (n x n)


for n in range(7,23,5):
     for d in range(2,7,2):

        matrices = []

        while len(matrices) < 100:
            # Create the random matrix
            result = create_random_matrix(n, d)

            embed_matrix = np.full(shape=(2,2), fill_value= -1)
            np.fill_diagonal(embed_matrix, 0) #change diagonal value to 1/0 for SA
            embed_2x2(result, embed_matrix, len(result)-2, len(result[1])-2)  

            G = nx.DiGraph(result)
            if nx.is_connected(G.to_undirected()):
                matrices.append(result)

        matrices = np.array(matrices)

        np.save(f'{n}N-{d}D', matrices)
# %%
