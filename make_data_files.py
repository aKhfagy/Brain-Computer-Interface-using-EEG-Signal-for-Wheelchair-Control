from datasetreading import motor_imaginary, segment_motor_data, get_features_motor_dataset, TUARv2

f5, gen = motor_imaginary(index=[4, 5])
index = [1, 2, 3, 4, 5]
mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
seg_f5 = segment_motor_data(f5, index, mapping, 'features.motor_dataset/seg_data_f5_c3_c4.npy')
seg_gen = segment_motor_data(gen, index, mapping, 'features.motor_dataset/seg_data_gen_c3_c4.npy')
get_features_motor_dataset(seg_f5, set_labels=index,
                           path_features='features.motor_dataset/seg_data_features_f5_c3_c4.npy',
                           path_labels='features.motor_dataset/seg_data_labels_f5_c3_c4.npy',
                           path_set_labels='features.motor_dataset/seg_data_set_labels_f5_c3_c4.npy')

get_features_motor_dataset(seg_gen, set_labels=index,
                           path_features='features.motor_dataset/seg_data_features_gen_c3_c4.npy',
                           path_labels='features.motor_dataset/seg_data_labels_gen_c3_c4.npy',
                           path_set_labels='features.motor_dataset/seg_data_set_labels_gen_c3_c4.npy')
del f5, gen, index, mapping, seg_f5, seg_gen

TUARv2()

