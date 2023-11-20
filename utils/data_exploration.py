### check how many data points are in Data/folder_1,2,3/256/0,1,2,3,4

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 0: Neoplastic cells, 1: Inflammatory, 2: Connective/Soft tissue cells, 3: Dead Cells, 4: Epithelial
f1_num_0= glob.glob('Data/folder_1/50/0/*')
f1_num_1= glob.glob('Data/folder_1/50/1/*')
f1_num_2= glob.glob('Data/folder_1/50/2/*')
f1_num_3= glob.glob('Data/folder_1/50/3/*')
f1_num_4= glob.glob('Data/folder_1/50/4/*')

f2_num_0= glob.glob('Data/folder_2/50/0/*')
f2_num_1= glob.glob('Data/folder_2/50/1/*')
f2_num_2= glob.glob('Data/folder_2/50/2/*')
f2_num_3= glob.glob('Data/folder_2/50/3/*')
f2_num_4= glob.glob('Data/folder_2/50/4/*')

f3_num_0= glob.glob('Data/folder_3/50/0/*')
f3_num_1= glob.glob('Data/folder_3/50/1/*')
f3_num_2= glob.glob('Data/folder_3/50/2/*')
f3_num_3= glob.glob('Data/folder_3/50/3/*')
f3_num_4= glob.glob('Data/folder_3/50/4/*')

# make a table row is the folder, column is the number of data points per cell type

columns = ['folder_1', 'folder_2', 'folder_3']
index = ['0', '1', '2', '3', '4']
data = [[len(f1_num_0), len(f2_num_0), len(f3_num_0)],
        [len(f1_num_1), len(f2_num_1), len(f3_num_1)],
        [len(f1_num_2), len(f2_num_2), len(f3_num_2)],
        [len(f1_num_3), len(f2_num_3), len(f3_num_3)],
        [len(f1_num_4), len(f2_num_4), len(f3_num_4)]]

df = pd.DataFrame(data, columns = columns, index = index)

df.index.name = 'cell_type'
df.index = ['Neoplastic cells', 'Inflammatory', 'Connective/Soft tissue cells', 'Dead Cells', 'Epithelial']

df.to_csv('results/data_points.csv')

