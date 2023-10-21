#%%
import multiprocessing
import time
from NetworkModelling.Embedmatrix_3a import *
#%%
if __name__ == '__main__':
    for node in range(7,9,5):
        for density in range(2,3,2):
            start = time.time()
            list_of_percent_pure_states = []
            pool = multiprocessing.Pool(processes=1)
            parameter = [(density, node)]*10
            list_of_percent_pure_states = pool.starmap(simulate, parameter)
            pool.close()
            pool.join()
            print(list_of_percent_pure_states)
            list_of_percent_pure_states = np.array(list_of_percent_pure_states)
            np.savetxt(f'{node}N_{density}d_Sub64_100_each_state.csv', list_of_percent_pure_states, delimiter=',')
            end = time.time()
            print(f"Time Elapsed:{end - start} ")
# %%
