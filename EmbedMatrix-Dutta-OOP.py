#%%
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import numpy as np

#A function to randomly choose state
def r():
    return random.choice([-1,1])

#Define the class network
class network:
    def __init__(self):
        a = r()
        b = r()
        self.nodes = [a,b]
        self.edge_matrix = []     

    def __str__(self):
        node_str = ''
        for i in self.nodes:
            if i == 1:
                node_str += '1'
            if i == -1:
                node_str += '0'
        return node_str

    def generate_nodes(self,n):
        for i in range(n):
            self.nodes.append(r())

    def set_nodes(self,l:list):
        self.nodes = l
    
    def rand_initial(self):
        """Randomly sets initial states for all nodes"""
        for i in range(len(self.nodes)):
            self.nodes[i] = r()

    def set_edges(self,l:list):
        self.edge_matrix = l

    def count_nodes(self):
        """Counts nodes other than the embedded toggle switch"""
        return len(self.nodes) - 2

    def count_edges(self):
        """Counts edges other than those of the embedded toggle switch"""
        e = 0
        for j in self.edge_matrix:
            for n in j:
                e += abs(n)
        return e - 2

    def generate_edges(self,edge_ratio):
        """Generates a fixed number of edges randomly in addtion to those of the embedded toggle switch"""
        
        node_count = self.count_nodes()

        for j,k in enumerate(self.nodes):
            self.edge_matrix.append([])
            for m,n in enumerate(self.nodes):
                self.edge_matrix[j].append(0)
        
        self.edge_matrix[0][1] = -1
        self.edge_matrix[1][0] = -1
        
        if node_count != 0:
            x = random.randint(2,1+node_count)
            y = random.choice([0,1])
            z = random.choice([0,1])

            if z == 0:
                self.edge_matrix[x][y] = random.choice([1,-1])
                fixed = (x,y)
            else:
                self.edge_matrix[y][x] = random.choice([1,-1])
                fixed = (y,x)

            while self.count_edges() < node_count*edge_ratio:
                p = random.randint(0,len(self.nodes)-1)
                q = random.randint(0,len(self.nodes)-1)
                exceptions = [(0,1),(1,0),fixed]
                if p != q and (p,q) not in exceptions:
                    self.edge_matrix[p][q] = random.choice([1,-1])
                    exceptions.append((p,q))

    def get_edges(self):
        print(self.edge_matrix)

    def update(self,n):
        """Updates the given node according to the Ising Model"""
        x = 0
        for i,j in enumerate(self.edge_matrix[n]):
            x += self.nodes[i]*j

        if x > 0:
            self.nodes[n] = 1
        elif x < 0:
            self.nodes[n] = -1

    def steady_state(self):
        """Returns the steady state of the network wrt the GA update scheme"""

        updated = []
        while len(updated) != len(self.nodes):
            n = random.randint(0,len(self.nodes)-1)
            if n not in updated:
                updated.append(n)
            self.update(n)
            # print(n,self)

        c = 0
        while c<30:
            x = self.nodes
            n = random.randint(0,len(self.nodes)-1)
            self.update(n)
            # print(n,self)
            y = self.nodes
            if x == y:
                c +=1
            else:
                c = 0

    def toggle_switch(self):
        self.generate_edges(0)

    def toggle_triad(self):
        self.generate_nodes(1)
        self.set_edges([[0,-1,-1],[-1,0,-1],[-1,-1,0]])

#Run a simulation of a single network
def runsim(N,t:int) -> dict:
    """Simulates given network, for the specified number of trials"""
    freq = {}
    c = 0
    for i in range(t):
        N.rand_initial()
        N.steady_state()
        
        p = str(N)
        if p[0:2] == '10' or p[0:2] == '01':
            c +=1
    #     if p in freq:
    #         freq[p] += 1/t
    #     else:
    #         freq[p] = 1/t

    # l = list(freq.keys())
    # l.sort()
    # freq1 = {i:freq[i] for i in l}

    return c/t

# def dictplot(freq:dict):
#     "Plots bar graphs from a dictionary"
#     plt.bar(range(len(freq)), list(freq.values()), align='center', color='darkorchid')
#     plt.xticks(range(len(freq)), list(freq.keys()), rotation = 90)
#     plt.subplots_adjust(bottom = 0.2)
#     plt.show()

#Repeat simulation for random networks of given parameters
def repsim(n,e,trials):
    F1data = []
    for i in range(trials):
        N = network()
        N.generate_nodes(n)
        N.generate_edges(e)
        F1 = runsim(N,1000)
        F1data.append(F1)
    F1data.sort()

    return F1data

#Obtaining Dataset
N5E2 = repsim(5,2,100)
N5E4 = repsim(5,4,100)
N5E6 = repsim(5,6,100)
data5 = [N5E2,N5E4,N5E6]

N10E2 = repsim(10,2,100)
N10E4 = repsim(10,4,100)
N10E6 = repsim(10,6,100)
data10 = [N10E2,N10E4,N10E6]

N15E2 = repsim(15,2,100)
N15E4 = repsim(15,4,100)
N15E6 = repsim(15,6,100)
data15 = [N15E2,N15E4,N15E6]

N20E2 = repsim(20,2,100)
N20E4 = repsim(20,4,100)
N20E6 = repsim(20,6,100)
data20 = [N20E2,N20E4,N20E6]

data = [N5E2,N5E4,N5E6,N10E2,N10E4,N10E6,N15E2,N15E4,N15E6,N20E2,N20E4,N20E6]

#Plotting
fig, ax = plt.subplots(figsize=(10, 6))
group_spacing = 3  
positions = np.arange(1, (len(data) // 3) * 4 + 1, step=4)

boxplots = []

for i in range(0, len(data), 3):
    if i + 2 < len(data):
        box1 = data[i]
        box2 = data[i + 1]
        box3 = data[i + 2]

        box_width = 0.4  

        boxplot1 = ax.boxplot(box1, positions=[positions[i//3] - box_width], widths=box_width, patch_artist=True, boxprops=dict(facecolor='darkcyan'))
        boxplot2 = ax.boxplot(box2, positions=[positions[i//3]], widths=box_width, patch_artist=True, boxprops=dict(facecolor='maroon'))
        boxplot3 = ax.boxplot(box3, positions=[positions[i//3] + box_width], widths=box_width, patch_artist=True, boxprops=dict(facecolor='seagreen'))

        boxplots.extend([boxplot1, boxplot2, boxplot3])

# medians = [np.median(data[i:i+3]) for i in range(0, len(data), 3)]
# ax.scatter(positions, medians, marker='o', color='black', zorder=5)

ax.set_title('Boxplot of Network Simulations')
ax.set_xlabel('Order of the Network')
ax.set_ylabel('Fraction of Canonical States(F1)')

ax.set_xticks(positions)
ax.set_xticklabels(['5N','10N','15N','20N'])

plt.legend(['E:2N','E:4N','E:6N'],loc = 'lower right')
plt.show()




    

# %%
