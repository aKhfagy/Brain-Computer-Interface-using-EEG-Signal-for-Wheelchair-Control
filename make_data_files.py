from datasetreading import motor_imaginary, get_features_motor_dataset, get_segmented_data_for_rnn, \
    TUARv2, wavelet_processing


data = motor_imaginary()
index = [1, 2, 3, 4, 5, 6, 91, 92, 99]
subjects = ['A', 'B', 'C', 'D', 'E', 'F']
for subject in subjects:
    get_features_motor_dataset(data[subject], set_labels=index,
                               path_features='features.motor_dataset/data_features' + subject + '.npy',
                               path_labels='features.motor_dataset/data_labels' + subject + '.npy',
                               path_set_labels='features.motor_dataset/data_set_labels' + subject + '.npy')

del index, data, subjects

#TUARv2()
