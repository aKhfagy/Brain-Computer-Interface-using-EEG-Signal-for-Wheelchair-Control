from FNN import FNN
from RandomForest import RF
from KNN import KNN
from datasets import TUARv2, load_processed_features_TUARv2, motor_imaginary

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

data = motor_imaginary()

print('Motor imaginary Data end')
print('==============================================================================================================')
