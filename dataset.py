import os
import numpy as np
import tensorflow as tf
from scipy.integrate import cumtrapz 


def synthetic_dataset():
    dataset = np.genfromtxt('synthetic_dataset/data_norm_200.csv', delimiter=',', dtype=np.float32).T
    _min = np.min(dataset)
    _max = np.max(dataset)
    dataset = (dataset - _min) / (_max - _min)
    return dataset


def real_dataset():
    data = np.genfromtxt('real_dataset/CWTcoefmean.csv', delimiter=',', dtype=np.float32)
    n_features = 10
    trapz = int(np.floor(data.shape[0] / n_features) * n_features)
    new_data = np.zeros((n_features, data.shape[1]))
    trapz_per_feature = int(trapz / n_features)
    for j in range(data.shape[1]):
        integ = cumtrapz(data[:, j])[-trapz - 1:]
        for i in range(n_features):
            new_data[i, j] = integ[trapz_per_feature * (i + 1)] - integ[trapz_per_feature * i]
    new_data = new_data / trapz_per_feature
    dataset = new_data.T[-200:]
    _min = np.min(dataset)
    _max = np.max(dataset)
    dataset = (dataset - _min) / (_max - _min)
    return dataset


def aramis_dataset_coea(train_path):
    train_list = os.listdir(train_path)
    data = np.genfromtxt(os.path.join(train_path, train_list[0]), delimiter=',', dtype=np.float32)[:, :-1]
    _max = np.max(data)
    _min = np.min(data)
    data = ((data - _min) / (_max - _min))
    return data[-200:, :]

def aramis_dataset(train_path, test_path, valid_ratio=0.2, batch_size=32):
    train_files = os.listdir(train_path)
    test_files = os.listdir(test_path)
    train_filepaths = []
    for file in train_files:
        train_filepaths.append(os.path.join(train_path, file))
    valid_dataset = tf.data.Dataset.list_files(train_filepaths[:int(valid_ratio*len(train_files))])
    train_dataset = tf.data.Dataset.list_files(train_filepaths[int(valid_ratio*len(train_files)):])
    test_filepaths = []
    for file in test_files:
        test_filepaths.append(os.path.join(test_path, file))
    test_dataset = tf.data.Dataset.list_files(test_filepaths)
    # return batched tuples of data and labels for each
    return train_dataset, valid_dataset, test_dataset
