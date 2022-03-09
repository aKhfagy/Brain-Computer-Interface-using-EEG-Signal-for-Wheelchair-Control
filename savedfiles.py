import numpy as np


def load_processed_features_TUARv2(path_features):
    features = np.load(path_features)
    labels = np.load('features.tuar/labels.npy')
    n_output = set(labels)
    n_output = len(n_output)

    return features, labels, n_output


def load_raw_motor_dataset_data(path_f5=None, path_gen=None):
    data_f5 = None
    data = None
    if path_f5 is not None:
        data_f5 = np.load(path_f5, allow_pickle=True)
        print('loaded F5 data')
    if path_gen is not None:
        data = np.load(path_gen, allow_pickle=True)
        print('loaded general data')

    return data_f5, data


def load_seg_motor_dataset(path_f5=None, path_gen=None):
    seg_f5 = None
    seg_gen = None
    if path_f5 is not None:
        seg_f5 = np.load(path_f5, allow_pickle=True)
        print('loaded segmented F5 data')
    if path_gen is not None:
        seg_gen = np.load(path_gen, allow_pickle=True)
        print('loaded segmented general data')

    return seg_f5, seg_gen


def load_features_motor_dataset(path_features, path_labels, path_set_labels):
    features = np.load(path_features)
    labels = np.load(path_labels)
    n_output = np.load(path_set_labels)
    n_output = len(n_output)
    return features, labels, n_output

