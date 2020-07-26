import numpy as np
from scipy.io import loadmat

data = loadmat('coeffmean-1B3x.mat')['CWTcoefmean']
np.savetxt('CWTcoefmean.csv', data, delimiter=',')