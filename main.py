from KNN import KNN
from SVM import SVM
from MLP import MLP
from RandomForest import RF
from savedfiles import load_processed_features_TUARv2, load_features_motor_dataset, load_motor_dataset

print('==============================================================================================================')
print('TUARv2 Data start')

features, labels, n_output = load_processed_features_TUARv2('features.tuar/features_mean_std_fp1_fp2.npy')

model, accuracy, error = KNN(features, labels, n_output)
print('KNN\nAccuracy: ', accuracy, ', Error: ', error)

model, accuracy, error = RF(features, labels)
print('Random Forest\nAccuracy: ', accuracy, ', Error: ', error)

model, accuracy, error = SVM(features, labels)
print('SVM\nAccuracy: ', accuracy, ', Error: ', error)

model, accuracy, error = MLP(features, labels)
print('MLP\nAccuracy: ', accuracy, ', Error: ', error)

del features, labels, n_output, model, accuracy, error

print('TUARv2 Data end')
print('==============================================================================================================')
print('==============================================================================================================')
print('Motor imaginary Data start')

print('Features as X')
features, labels, n_outputs = load_features_motor_dataset('features.motor_dataset/data_features.npy',
                                                          'features.motor_dataset/data_labels.npy',
                                                          'features.motor_dataset/data_set_labels.npy')

model, accuracy, error = KNN(features, labels, n_outputs)
print('KNN\nAccuracy: ', accuracy, ', Error: ', error)

model, accuracy, error = RF(features, labels)
print('Random Forest\nAccuracy: ', accuracy, ', Error: ', error)

model, accuracy, error = SVM(features, labels)
print('SVM\nAccuracy: ', accuracy, ', Error: ', error)

model, accuracy, error = MLP(features, labels)
print('MLP\nAccuracy: ', accuracy, ', Error: ', error)

del features, labels, model, accuracy, error, n_outputs

print('Motor imaginary Data end')
print('==============================================================================================================')
