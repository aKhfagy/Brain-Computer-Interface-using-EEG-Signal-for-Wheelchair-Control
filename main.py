from FNN import FNN
from KNN import KNN
from RandomForest import RF
from datasets import TUARv2, load_processed_features_TUARv2, motor_imaginary, load_raw_motor_dataset_data, \
    segment_motor_data, get_features_motor_dataset, load_features_motor_dataset

# print('==============================================================================================================')
# print('TUARv2 Data start')
#
# features, labels, n_output = load_processed_features_TUARv2()
#
# fuzzy = FNN(features, labels)
# fuzzy.make_model(n_inputs=4, n_hidden=4, n_outputs=n_output, n_iterations=100)
#
# knn, knn_accuracy, knn_error = KNN(features, labels, n_output)
# print('KNN\nAccuracy: ', knn_accuracy, ', Error: ', knn_error)
#
# rf, rf_accuracy, rf_error = RF(features, labels)
# print('Random Forest\nAccuracy: ', rf_accuracy, ', Error: ', rf_error)
#
# del features, labels, n_output, fuzzy, knn, knn_accuracy, knn_error, rf, rf_accuracy, rf_error
#
# print('TUARv2 Data end')
# print('==============================================================================================================')
print('==============================================================================================================')
print('Motor imaginary Data start')

data_f5, data = load_raw_motor_dataset_data()

LABELS_F5 = [1, 2, 3, 4, 5]
MAP_F5 = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
seg_f5 = segment_motor_data(data_f5, LABELS_F5, MAP_F5)
del data_f5, LABELS_F5, MAP_F5

LABELS_GENERAL = [1, 2, 3, 4, 5, 6]
MAP_GENERAL = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
seg_gen = segment_motor_data(data, LABELS_GENERAL, MAP_GENERAL)
del data, LABELS_GENERAL, MAP_GENERAL

print('Motor imaginary Data end')
print('==============================================================================================================')
