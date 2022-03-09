from FNN import FNN
from KNN import KNN
from RandomForest import RF
from savedfiles import load_processed_features_TUARv2, load_features_motor_dataset, \
    load_seg_motor_dataset, load_raw_motor_dataset_data

print('==============================================================================================================')
print('TUARv2 Data start')

features, labels, n_output = load_processed_features_TUARv2()

fuzzy = FNN(features, labels)
fuzzy.make_model(n_inputs=4, n_hidden=4, n_outputs=n_output, n_iterations=100)

knn, knn_accuracy, knn_error = KNN(features, labels, n_output)
print('KNN\nAccuracy: ', knn_accuracy, ', Error: ', knn_error)

rf, rf_accuracy, rf_error = RF(features, labels)
print('Random Forest\nAccuracy: ', rf_accuracy, ', Error: ', rf_error)

del features, labels, n_output, fuzzy, knn, knn_accuracy, knn_error, rf, rf_accuracy, rf_error

print('TUARv2 Data end')
print('==============================================================================================================')
print('==============================================================================================================')
print('Motor imaginary Data start')

print('F5 data start')
features_f5, labels_f5, n_outputs_f5 = load_features_motor_dataset('features.motor_dataset/features_f5_mean_std.npy',
                                                                   'features.motor_dataset/labels_f5.npy',
                                                                   'features.motor_dataset/set_f5.npy')

fuzzy = FNN(features_f5, labels_f5)
fuzzy.make_model(n_inputs=4, n_hidden=4, n_outputs=n_outputs_f5, n_iterations=100)

knn, knn_accuracy, knn_error = KNN(features_f5, labels_f5, n_outputs_f5)
print('KNN\nAccuracy: ', knn_accuracy, ', Error: ', knn_error)

rf, rf_accuracy, rf_error = RF(features_f5, labels_f5)
print('Random Forest\nAccuracy: ', rf_accuracy, ', Error: ', rf_error)

del features_f5, labels_f5, n_outputs_f5, fuzzy, knn, knn_accuracy, knn_error, rf, rf_accuracy, rf_error
print('F5 data end')

print('General data start')
features_gen, labels_gen, n_outputs_gen = load_features_motor_dataset(
    'features.motor_dataset/features_gen_mean_std.npy',
    'features.motor_dataset/labels_gen.npy',
    'features.motor_dataset/set_gen.npy')

fuzzy = FNN(features_gen, labels_gen)
fuzzy.make_model(n_inputs=4, n_hidden=4, n_outputs=n_outputs_gen, n_iterations=100)

knn, knn_accuracy, knn_error = KNN(features_gen, labels_gen, n_outputs_gen)
print('KNN\nAccuracy: ', knn_accuracy, ', Error: ', knn_error)

rf, rf_accuracy, rf_error = RF(features_gen, labels_gen)
print('Random Forest\nAccuracy: ', rf_accuracy, ', Error: ', rf_error)

del features_gen, labels_gen, n_outputs_gen, fuzzy, knn, knn_accuracy, knn_error, rf, rf_accuracy, rf_error
print('General data end')


print('Motor imaginary Data end')
print('==============================================================================================================')
