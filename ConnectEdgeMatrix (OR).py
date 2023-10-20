import numpy as np
import networkx as nx

def generate_matrix(n, d):
    matrix = np.zeros((n, n), dtype=int)
    indices = []
    while len(indices) < n*d:
        indices = np.random.choice(n*n, n*d, replace=False) 
        # Ensure no element on the diagonal gets updated #Clever Idea!
        indices = [idx for idx in indices if idx % (n+1) != 0]

    matrix.flat[indices] = np.random.choice([-1, 1], len(indices))
    return matrix

# Function to embed a 2x2 matrix
def embed_2x2(matrix, sub_matrix, row, col):
        matrix[row:row+2, col:col+2] = sub_matrix

node = 5 # Change
density = 2 # Change
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