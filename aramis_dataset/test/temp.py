import os
import numpy as np
from scipy.io import loadmat

test = loadmat('test.mat')['Test_data']
os.mkdir('data_csv')

#%%
for s in range(np.size(test)):
    os.mkdir('data_csv/system_{}'.format(s + 1))
    sys = test[0, s]
    for c in range(np.shape(sys['trajectory'])[1]):
        os.mkdir('data_csv/system_{}/component_{}'.format((s + 1), (c + 1)))
        data = sys['trajectory'][0,c]
        np.savetxt('data_csv/system_{}/component_{}/data.csv'\
                   .format((s + 1), (c + 1)),
                   data, delimiter=',')
