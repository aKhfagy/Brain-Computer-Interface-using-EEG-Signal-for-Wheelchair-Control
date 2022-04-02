from KNN import KNN
from SVM import SVM
from MLP import MLP
from RandomForest import RF
from savedfiles import load_processed_features_TUARv2, load_features_motor_dataset
from joblib import dump

print('==============================================================================================================')
print('TUARv2 Data start')

features, labels, n_output = load_processed_features_TUARv2('features.tuar/features_mean_std_fp1_fp2.npy')

model, accuracy, error = KNN(features, labels, n_output)
dump(model, 'models.tuar/KNN.joblib')
print('KNN\nAccuracy: ', accuracy, ', Error: ', error)

model, accuracy, error = RF(features, labels)
dump(model, 'models.tuar/RF.joblib')
print('Random Forest\nAccuracy: ', accuracy, ', Error: ', error)

model, accuracy, error = SVM(features, labels)
dump(model, 'models.tuar/SVM.joblib')
print('SVM\nAccuracy: ', accuracy, ', Error: ', error)

model, accuracy, error = MLP(features, labels)
dump(model, 'models.tuar/MLP.joblib')
print('MLP\nAccuracy: ', accuracy, ', Error: ', error)

del features, labels, n_output, model, accuracy, error

print('TUARv2 Data end')
print('==============================================================================================================')
print('==============================================================================================================')
print('Motor imaginary Data start')

subjects = ['A', 'B', 'C', 'D', 'E', 'F']

for subject in subjects:
    print('Subject:', subject)
    features, labels, n_outputs = load_features_motor_dataset('features.motor_dataset/data_features'
                                                              + subject + '.npy',
                                                              'features.motor_dataset/data_labels'
                                                              + subject + '.npy',
                                                              'features.motor_dataset/data_set_labels'
                                                              + subject + '.npy')
    model, accuracy, error = KNN(features, labels, n_outputs)
    dump(model, 'models.cla/' + subject + '-KNN.joblib')
    print('KNN\nAccuracy:', accuracy, ', Error:', error)

    model, accuracy, error = RF(features, labels)
    dump(model, 'models.cla/' + subject + '-RF.joblib')
    print('Random Forest\nAccuracy:', accuracy, ', Error:', error)

    model, accuracy, error = SVM(features, labels)
    dump(model, 'models.cla/' + subject + '-SVM.joblib')
    print('SVM\nAccuracy:', accuracy, ', Error:', error)

    model, accuracy, error = MLP(features, labels)
    dump(model, 'models.cla/' + subject + '-MLP.joblib')
    print('MLP\nAccuracy:', accuracy, ', Error:', error)

    del features, labels, model, accuracy, error, n_outputs

print('Motor imaginary Data end')
print('==============================================================================================================')
