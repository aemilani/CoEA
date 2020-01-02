import os
import numpy as np
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


def aramis_dataset_classification():
    train_path = 'aramis_dataset/train/data_csv/'
    test_path = 'aramis_dataset/test/data_csv/'
    n_signals = 10
    train_data, test_data, train_labels = [], [], []
    for i in range(n_signals):
        train_data.append([])
        test_data.append([])
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels)
    train_systems = os.listdir(train_path)
    test_systems = os.listdir(test_path)
    train_systems.sort(key=lambda x: int(x[7:]))
    test_systems.sort(key=lambda x: int(x[7:]))
    for syst in train_systems:
        syst_path = train_path + syst + '/'
        components = os.listdir(syst_path)
        for comp in components:
            comp_path = syst_path + comp + '/'
            data = np.genfromtxt(comp_path + 'data.csv', delimiter=',', dtype=np.float32)
            labels = np.genfromtxt(comp_path + 'labels.csv', delimiter=',', dtype=np.float32)
            train_data = np.concatenate((train_data, data), axis=1)
            train_labels = np.concatenate((train_labels, labels))
    for syst in test_systems:
        syst_path = test_path + syst + '/'
        components = os.listdir(syst_path)
        for comp in components:
            comp_path = syst_path + comp + '/'
            data = np.genfromtxt(comp_path + 'data.csv', delimiter=',', dtype=np.float32)
            test_data = np.concatenate((test_data, data), axis=1)
    all_data = np.concatenate((train_data, test_data), axis=1)
    _max = np.max(all_data)
    _min = np.min(all_data)
    train_data = (train_data - _min) / (_max - _min)
    test_data = (test_data - _min) / (_max - _min)
    return train_data.T, train_labels, test_data.T


def aramis_dataset_coea():
    train_path = 'aramis_dataset/train/data_csv/'
    train_systems = os.listdir(train_path)
    train_systems.sort(key=lambda x: int(x[7:]))
    syst_path = train_path + train_systems[0] + '/'
    components = os.listdir(syst_path)
    comp_path = syst_path + components[0] + '/'
    data = np.genfromtxt(comp_path + 'data.csv', delimiter=',', dtype=np.float32)
    _max = np.max(data)
    _min = np.min(data)
    data = ((data - _min) / (_max - _min)).T
    return data[-200:, :]
