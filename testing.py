import numpy as np
from RNN import compile_model
from keras import backend as K
from keras.utils import np_utils
from savedfiles import load_seg_data_deep_motor_dataset

f5, labels, n_outputs = \
    load_seg_data_deep_motor_dataset(path_data='features.motor_dataset/seg_data_deep_f5_c3_c4.npy',
                                     path_labels='features.motor_dataset/seg_data_labels_deep_f5_c3_c4.npy',
                                     path_set_labels='features.motor_dataset/seg_data_set_labels_deep_f5_c3_c4.npy')
f5 = f5.astype(np.float32)
labels = np_utils.to_categorical(labels, n_outputs)
labels = np.expand_dims(labels, -1)
x = K.cast_to_floatx(f5)
y = K.cast_to_floatx(labels)
print('RNN')
rnn = compile_model(x, y, f5.shape[1], f5.shape[2], 'F5_RNN')



