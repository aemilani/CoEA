import numpy as np
from scipy.io import loadmat

test_taus = loadmat('test_data_groundtruth.mat')['Test_groundtruth']

np.savetxt('test_taus.csv', test_taus, delimiter=',')