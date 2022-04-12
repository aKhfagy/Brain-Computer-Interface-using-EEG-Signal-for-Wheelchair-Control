from KNN import knn
from SVM import svm
from MLP import mlp
from RandomForest import rf
from savedfiles import load_features_motor_dataset
from joblib import dump

print('==============================================================================================================')
print('CLA Data start')

subjects = ['A', 'B', 'C', 'D', 'E', 'F']

for subject in subjects:
    print('Subject:', subject)
    features, labels, n_outputs = load_features_motor_dataset('features.motor_dataset/data_features'
                                                              + subject + '.npy',
                                                              'features.motor_dataset/data_labels'
                                                              + subject + '.npy',
                                                              'features.motor_dataset/data_set_labels'
                                                              + subject + '.npy')
    model, accuracy, error = knn(features, labels, n_outputs)
    dump(model, 'models.cla/' + subject + '-KNN.joblib')
    print('KNN\nAccuracy:', accuracy, ', Error:', error)

    model, accuracy, error = rf(features, labels)
    dump(model, 'models.cla/' + subject + '-RF.joblib')
    print('Random Forest\nAccuracy:', accuracy, ', Error:', error)

    model, accuracy, error = svm(features, labels)
    dump(model, 'models.cla/' + subject + '-SVM.joblib')
    print('SVM\nAccuracy:', accuracy, ', Error:', error)

    model, accuracy, error = mlp(features, labels)
    dump(model, 'models.cla/' + subject + '-MLP.joblib')
    print('MLP\nAccuracy:', accuracy, ', Error:', error)

    del features, labels, model, accuracy, error, n_outputs

print('CLA Data end')
print('==============================================================================================================')
