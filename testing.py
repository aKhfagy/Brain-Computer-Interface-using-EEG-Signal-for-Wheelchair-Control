from RNN import compile_model
from savedfiles import load_seg_data_deep_motor_dataset

f5, labels, n_outputs = \
    load_seg_data_deep_motor_dataset(path_data='features.motor_dataset/seg_data_deep_f5_c3_c4.npy',
                                     path_labels='features.motor_dataset/seg_data_labels_deep_f5_c3_c4.npy',
                                     path_set_labels='features.motor_dataset/seg_data_set_labels_deep_f5_c3_c4.npy')
print('RNN')
rnn = compile_model(f5.shape[2], f5.shape[1], 'F5_RNN')



