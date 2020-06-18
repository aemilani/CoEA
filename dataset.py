import os
import numpy as np
from scipy.integrate import cumtrapz


def synthetic_dataset():
    dataset = np.genfromtxt('synthetic_dataset/data_norm_200.csv',
                            delimiter=',', dtype=np.float32).T
    _min = np.min(dataset)
    _max = np.max(dataset)
    dataset = (dataset - _min) / (_max - _min)
    return dataset


def real_dataset():
    data = np.genfromtxt('real_dataset/CWTcoefmean.csv',
                         delimiter=',', dtype=np.float32)
    n_features = 10
    trapz = int(np.floor(data.shape[0] / n_features) * n_features)
    new_data = np.zeros((n_features, data.shape[1]))
    trapz_per_feature = int(trapz / n_features)
    for j in range(data.shape[1]):
        integ = cumtrapz(data[:, j])[-trapz - 1:]
        for i in range(n_features):
            new_data[i, j] = integ[trapz_per_feature * (i + 1)] - \
                             integ[trapz_per_feature * i]
    new_data = new_data / trapz_per_feature
    dataset = new_data.T[-200:]
    _min = np.min(dataset)
    _max = np.max(dataset)
    dataset = (dataset - _min) / (_max - _min)
    return dataset


def aramis_dataset(train_path='aramis_dataset/train/data_csv',
                   test_path='aramis_dataset/test/data_csv'):
    train_files = os.listdir(train_path)
    train_files.sort()
    for i in range(len(train_files)):
        train_files[i] = os.path.join(train_path, train_files[i])
    test_files = os.listdir(test_path)
    test_files.sort()
    for i in range(len(test_files)):
        test_files[i] = os.path.join(test_path, test_files[i])
    _min, _max = 0, 0
    train_files_with_elbow_point = []
    for file in train_files:
        matrix = np.genfromtxt(file, dtype=np.float32, delimiter=',')
        features = matrix[:, :-1]
        labels = matrix[:, -1]
        if np.max(labels) == 1:
            if np.min(features) < _min:
                _min = np.min(features)
            if np.max(features) > _max:
                _max = np.max(features)
            train_files_with_elbow_point.append(file)
    for file in test_files:
        features = np.genfromtxt(file, dtype=np.float32, delimiter=',')
        if np.min(features) < _min:
            _min = np.min(features)
        if np.max(features) > _max:
            _max = np.max(features)
    return train_files_with_elbow_point, test_files, _min, _max


def normalize(data, min_value, max_value):
    """Normalizes the given data, so that the given min and max value correspond to 0, 1"""
    return (data - min_value) / (max_value - min_value)


def balanced_sample(data, rand_seed=None):
    np.random.seed(rand_seed)
    labels = data[:, -1]
    label, count = np.unique(labels, return_counts=True)
    label_counts = dict(zip(label, count))
    zeros_indices = np.sort(np.random.choice(list(range(label_counts[0])), size=label_counts[1], replace=False))
    zeros = data[zeros_indices, :]
    ones = data[-label_counts[1]:, :]
    return np.concatenate((zeros, ones), axis=0)
