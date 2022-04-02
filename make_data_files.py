from datasetreading import motor_imaginary, get_features_motor_dataset, get_segmented_data_for_rnn, \
    TUARv2, wavelet_processing


data = motor_imaginary()
index = [1, 2, 3, 4, 5, 6, 91, 92, 99]
get_features_motor_dataset(data, set_labels=index,
                           path_features='features.motor_dataset/data_features.npy',
                           path_labels='features.motor_dataset/data_labels.npy',
                           path_set_labels='features.motor_dataset/data_set_labels.npy')

del index, data

TUARv2()
