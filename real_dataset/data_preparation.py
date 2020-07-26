import os
import numpy as np


test1_folder = 'ims_bearing_dataset/1st_test'
test2_folder = 'ims_bearing_dataset/2nd_test'
test1_files = os.listdir(test1_folder)
test2_files = os.listdir(test2_folder)

test1_filepaths = []
for i in range(len(test1_files)):
    test1_filepaths.append(os.path.join(test1_folder, test1_files[i]))
test2_filepaths = []
for i in range(len(test2_files)):
    test2_filepaths.append(os.path.join(test2_folder, test2_files[i]))
    
#%%