import os
import logging
import numpy as np


def setup_logger(logger_name, log_path):
    log_file_path = os.path.join(log_path, '{}.log'.format(logger_name.lower()))
    file_format = logging.Formatter('[%(asctime)s] %(levelname)-8s %(message)s',
                                    datefmt='%Y/%m/%d %H:%M:%S')
    console_format = logging.Formatter('%(levelname)-8s %(message)s')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(file_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(console_format)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def calc_rho(indicators):
    if len(np.shape(indicators)) == 1:
        indicators = np.expand_dims(indicators, axis=-1)
    v = np.zeros(len(indicators.T))
    rho_mk = np.zeros(len(indicators.T))
    for i in range(len(indicators.T)):
        for k in range(len(indicators) - 1):
            for j in range(k + 1, len(indicators)):
                v[i] += np.sign(indicators[j][i] - indicators[k][i])
    n = len(indicators)
    sigma = np.sqrt(n * (n - 1) * (2 * n + 5) / 18)
    for i in range(len(indicators.T)):
        if v[i] > 1:
            rho_mk[i] = (v[i] - 1) / sigma
        elif v[i] == 0:
            rho_mk[i] = 0
        else:
            rho_mk[i] = (v[i] + 1) / sigma
    return rho_mk


def calc_delta(true_labels, predicted_labels):
    k_false, k_missed = 10000, 10000

    if np.max(true_labels) == 0:
        tau = np.nan
    else:
        tau = np.argmax(true_labels)

    if np.max(predicted_labels) == 0:
        tau_hat = np.nan
    else:
        tau_hat = np.argmax(predicted_labels)

    if (not np.isnan(tau)) and (not np.isnan(tau_hat)):
        delta = tau - tau_hat
    elif np.isnan(tau) and np.isnan(tau_hat):
        delta = 0
    elif np.isnan(tau) and (not np.isnan(tau_hat)):
        delta = k_false
    else:
        delta = - k_missed

    return delta


def aramis_metric(true_labels_list, predicted_labels_list):
    """Inputs are lists of numpy arrays."""
    deltas = []
    for true_label, predicted_label in zip(true_labels_list, predicted_labels_list):
        delta = calc_delta(true_label, predicted_label)
        deltas.append(delta)
    sum_phis = 0
    for delta in deltas:
        sum_phis += phi(delta)
    return sum_phis / len(deltas)


def phi(delta):
    a1 = 10
    a2 = 13
    b1 = 1 / (1 - np.exp(-1000 / a1))
    b2 = 1 / (1 - np.exp(-1000 / a2))
    if delta < -1000:
        return 1
    elif -1000 <= delta < 0:
        return (1 - np.exp(delta / a1)) * b1
    elif 0 <= delta <= 1000:
        return (1 - np.exp(-delta / a2)) * b2
    else:
        return 1


def true_test_labels(taus_path, test_files):
    taus = list(np.genfromtxt(taus_path, delimiter=','))
    labels = []
    for tau, file in zip(taus, test_files):
        features = np.genfromtxt(file, delimiter=',')
        n = np.shape(features)[0]
        if np.isnan(tau):
            label = [0] * n
            labels.append(np.array(label))
        else:
            label = [0] * int(tau-1)
            ones = [1] * int(n - (tau-1))
            label.extend(ones)
            labels.append(np.array(label))
    return labels


def labels_to_taus(labels):
    taus = []
    for label in labels:
        if np.max(label) == 0:
            tau = np.nan
        else:
            tau = np.argmax(label)
        taus.append(tau)
    return taus
