#%%
import os
import pandas as pd
import numpy as np
from numpy import array
import random 
import matplotlib.pyplot as plt
import seaborn as sns
#%%

# print(x)
nodes = ['GF', 'RTK', 'RAS', 'PI3K', 'RAF', 'PIP3', 'MEK_ERK', 'AKT', 'TF'] # , 'RTKi', 'MEKi'  add these for line graphs
# for adding weighted probablity for line graphs
# weights = [1] * 9
# weights[-1] = 0.2
def GF(x):
    x[0] = x[0]
    
def RTK(x):
    x[1] = x[0]
    # x[1] = (x[0] and not x[9]) with RTKinactivation
    # x[1] = 0 #for mek mutation and RAS mutation

def RAS(x):
    x[2] = x[1]
    # x[2] = 1 #for mek mutation
def PI3K(x):
    x[3] = (x[1] or x[2])

def RAF(x):
    x[4] = x[2]

def PIP3(x):
    x[5] = x[3]
    
def MEK_ERK(x):
    x[6] = x[4]
    # x[6] = x[4] and not x[10] with MTKinact

def AKT(x):
    x[7] = x[5]
    
def TF(x):
    x[8] = x[7] and x[6]

# def RTKi(x):
#     x[9] = 1

# def RAFmut(x):
#     x[2] = 1

# def MEKi(x):
#     x[10] = 1

#%%
func = [GF, RTK, RAS, PI3K, RAF, PIP3, MEK_ERK, AKT, TF]

def singlerun():

    x_all = []
    x = [0] * 9
    x[0] = 1
    #x[2] = 1#for mek mutation
    xsum = sum(x)
    # print(xsum)
    x_all.append(list(x))
    for i in range(120):
        # if i >=20:
        #     # RTKi(x)
        #    # RAFmut(x)
        #     # MEKi(x)
        # # f1 = random.choices(func, weights)
        f = random.choice(func)
        # # print(f1)
        # x_all.append(list(x))
        f(x)

        #for heatmaps
        if sum(x) > xsum:
            x_all.append(list(x))
            xsum = sum(x) 
        else:    
            continue
    print(x_all)
    return x_all


#%%
#for heatmaps
h = singlerun()
x_all = array(h)
print(x_all)
sns.heatmap(x_all, cmap='coolwarm', linewidths=0.5, xticklabels= nodes)
plt.show()


#%%
#Running simulation
x_simu = []
for i in range(10000):
    x_simu.append(singlerun())

b = array(x_simu)
print(b.shape)

merged_matrix = array(np.mean(b, axis=0))
print(merged_matrix.shape)
print(merged_matrix)

#merged_matrix.plot
plt.plot(merged_matrix, label = nodes, linewidth = 2.5)
plt.legend()
plt.show()
#%%
plt.savefig("MEKi_Fig2D.png")

# %%
os.getcwd()
#%%
#os.mkdir('./Plots/')
os.chdir('./Plots')
# %%
