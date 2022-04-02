import numpy as np


def load_processed_features_TUARv2(path_features):
    features = np.load(path_features)
    labels = np.load('features.tuar/labels.npy')
    n_output = set(labels)
    n_output = len(n_output)

    return features, labels, n_output


def load_motor_dataset(path=None):
    seg = None
    if path is not None:
        seg = np.load(path, allow_pickle=True)
        print('loaded', path)

    return seg


def load_features_motor_dataset(path_features, path_labels, path_set_labels):
    features = np.load(path_features, allow_pickle=True)
    labels = np.load(path_labels)
    n_output = np.load(path_set_labels)
    n_output = len(n_output)
    return features, labels, n_output


def load_seg_data_deep_motor_dataset(path_data, path_labels, path_set_labels):
    data = np.load(path_data)
    labels = np.load(path_labels)
    n_output = np.load(path_set_labels)
    n_output = len(n_output)
    return data, labels, n_output

