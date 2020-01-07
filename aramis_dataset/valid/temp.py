import os
import numpy as np
from scipy.io import loadmat

train = loadmat('Train.mat')['Train_data']
os.mkdir('data_csv')

#%%
for s in range(int(np.size(train) * 0.75), np.size(train)):
    os.mkdir('data_csv/system_{}'.format(s + 1))
    sys = train[0, s]
    all_labels = sys['Label']
    for c in range(np.shape(all_labels)[0]):
        os.mkdir('data_csv/system_{}/component_{}'.format((s + 1), (c + 1)))
        data = sys['trajectory'][0, c]
        labels = all_labels[c, :]
        np.savetxt('data_csv/system_{}/component_{}/data.csv'\
                   .format((s + 1), (c + 1)),
                   data, delimiter=',')
        np.savetxt('data_csv/system_{}/component_{}/labels.csv'\
                   .format((s + 1), (c + 1)),
                   labels, delimiter=',')
