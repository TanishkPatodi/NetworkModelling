'''
takes the matrix as input and return if its connected or not and the graph of connection
'''
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Define the adjacency matrix
matrix = np.array([ [ 0,  0,  0,  0,  0,  0,  0],
                    [ 0,  0,  1, -1,  0,  0,  0],
                    [-1,  0,  0, -1, -1,  0,  0],
                    [ 0, -1,  0,  0,  0,  0,  0],
                    [ 0, -1,  0, -1,  0,  0,  0],
                    [ 0,  0, -1,  1,  0,  0,  0],
                    [ 0,  1,  0,  0,  0, -1,  0]])
# n = 7
# matrix = np.zeros((n, n), dtype=int)
# indices = np.random.choice(n*n, 12, replace=False)
# matrix.flat[indices] = np.random.choice([-1, 1], 12)
# print(matrix)
#%%
# Create a directed graph from the adjacency matrix
G = nx.DiGraph(matrix)

# Check if the graph is connected
is_connected = nx.is_connected(G.to_undirected())

# Print the result
print(f"Is connected: {is_connected}")

# Define edge colors (red for inhibition, green for activation)
edge_colors = ['red' if matrix[i][j] == -1 else 'green' for i, j in G.edges()]

# Draw the graph
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=100)
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', font_weight='bold')
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrowsize=20)

# Display the plot
plt.title("Random Regulatory Network")
plt.show()

# %%
