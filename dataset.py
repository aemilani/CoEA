import os
import numpy as np
from scipy.integrate import cumtrapz
from keras.utils import Sequence


class AramisSequence(Sequence):
    """Creates a keras Sequence of the data related to the systems contained in the given path.
    Yields numpy arrays of component data normalized in (0, 1), and labels, one component at a time.
    """
    def __init__(self, path):
        x_set, y_set = [], []
        systems = os.listdir(path)
        systems.sort(key=lambda x: int(x[7:]))
        _min = 0.
        _max = 0.
        for system in systems:
            system_path = os.path.join(path, system)
            components = os.listdir(system_path)
            for component in components:
                component_path = os.path.join(system_path, component)
                data_path = os.path.join(component_path, 'data.csv')
                labels_path = os.path.join(component_path, 'labels.csv')
                data = np.genfromtxt(data_path , delimiter=',', dtype=np.float32)
                labels = np.genfromtxt(labels_path, delimiter=',', dtype=np.float32)
                x_set.append(data.T)
                y_set.append(labels)
                _min = min(_min, np.min(data))
                _max = max(_max, np.max(data))
        for i in range(len(x_set)):
            x_set[i] = (x_set[i] - _min) / (_max - _min)
        self.x = x_set
        self.y = y_set

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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
    train_systems = os.listdir(train_path)
    train_systems.sort(key=lambda x: int(x[7:]))
    syst_path = os.path.join(train_path, train_systems[0])
    components = os.listdir(syst_path)
    comp_path = os.path.join(syst_path, components[0])
    data = np.genfromtxt(os.path.join(comp_path, 'data.csv'), delimiter=',', dtype=np.float32)
    _max = np.max(data)
    _min = np.min(data)
    data = ((data - _min) / (_max - _min)).T
    return data[-200:, :]


def aramis_dataset_classification(train_path, valid_path):
    train_generator = AramisSequence(train_path)
    valid_generator = AramisSequence(valid_path)
    train_x, train_y = train_generator.x, train_generator.y
    valid_x, valid_y = valid_generator.x, valid_generator.y
    train_data, train_labels, valid_data, valid_labels = train_x[0], train_y[0], valid_x[0], valid_y[0]
    for i in range(len(train_x) - 1):
        train_data = np.concatenate((train_data, train_x[i+1]))
        train_labels = np.concatenate((train_labels, train_y[i + 1]))
    for i in range(len(valid_x) - 1):
        valid_data = np.concatenate((valid_data, valid_x[i+1]))
        valid_labels = np.concatenate((valid_labels, valid_y[i + 1]))
    return train_data, train_labels, valid_data, valid_labels
