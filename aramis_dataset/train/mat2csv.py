import os
import numpy as np
from scipy.io import loadmat

train = loadmat('Train.mat')['Train_data']
os.mkdir('data_csv')

#%%
for s in range(np.size(train)):
    sys = train[0, s]
    all_labels = sys['Label']
    for c in range(np.shape(all_labels)[0]):
        data = sys['trajectory'][0, c]
        labels = all_labels[c, :]
        labels = np.expand_dims(labels, axis=0)
        data_and_labels = np.concatenate((data, labels))
        np.savetxt('data_csv/sys{}_comp{}.csv'\
                   .format(str(s + 1).zfill(3), (c + 1)),
                   data_and_labels, delimiter=',')
