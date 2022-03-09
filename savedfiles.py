import numpy as np


def load_processed_features_TUARv2(path_features='features.tuar/features_mean_std.npy'):
    features = np.load(path_features)
    labels = np.load('features.tuar/labels.npy')
    n_output = set(labels)
    n_output = len(n_output)

    return features, labels, n_output


def load_raw_motor_dataset_data():
    data_f5 = np.load('features.motor_dataset/raw_data_f5.npy', allow_pickle=True)
    print('loaded F5 data')
    data = np.load('features.motor_dataset/raw_data_general.npy', allow_pickle=True)
    print('loaded general data')

    return data_f5, data


def load_seg_motor_dataset():
    seg_f5 = np.load('features.motor_dataset/seg_data_f5.npy', allow_pickle=True)
    print('loaded segmented F5 data')
    seg_gen = np.load('features.motor_dataset/seg_data_gen.npy', allow_pickle=True)
    print('loaded segmented general data')
    return seg_f5, seg_gen


def load_features_motor_dataset(path_features, path_labels, path_set_labels):
    features = np.load(path_features)
    labels = np.load(path_labels)
    n_output = np.load(path_set_labels)
    n_output = len(n_output)
    return features, labels, n_output

