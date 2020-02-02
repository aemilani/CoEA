import os
import numpy as np
from scipy.io import loadmat

test = loadmat('test.mat')['Test_data']
os.mkdir('data_csv')

#%%
for s in range(np.size(test)):
    sys = test[0, s]
    for c in range(np.shape(sys['trajectory'])[1]):
        data = sys['trajectory'][0,c]
        np.savetxt('data_csv/sys{}_comp{}.csv'\
                   .format(str(s + 1).zfill(2), (c + 1)),
                   data, delimiter=',')