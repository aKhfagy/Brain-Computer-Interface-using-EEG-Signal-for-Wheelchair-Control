from KNN import knn
from MLP import mlp
from RandomForest import rf
from SVM import svm
from savedfiles import load_features_motor_dataset
from joblib import dump
from flask import jsonify


def choose_model_by_name(model_name, subject):
    features, labels, n_outputs = load_features_motor_dataset('features.motor_dataset/data_features'
                                                              + subject + '.npy',
                                                              'features.motor_dataset/data_labels'
                                                              + subject + '.npy',
                                                              'features.motor_dataset/data_set_labels'
                                                              + subject + '.npy')
    if model_name == 'KNN':
        model, accuracy, error = knn(features, labels, n_outputs)
        dump(model, 'models.cla/' + subject + '-KNN.joblib')
        return jsonify({
            'Success': 'Made model ' + model_name + ' for subject ' + subject,
            'Accuracy': str(accuracy),
            'Loss': str(error)
        })

    elif model_name == 'RF':
        model, accuracy, error = rf(features, labels)
        dump(model, 'models.cla/' + subject + '-RF.joblib')
        return jsonify({
            'Success': 'Made model ' + model_name + ' for subject ' + subject,
            'Accuracy': str(accuracy),
            'Loss': str(error)
        })

    elif model_name == 'SVM':
        model, accuracy, error = svm(features, labels)
        dump(model, 'models.cla/' + subject + '-SVM.joblib')
        return jsonify({
            'Success': 'Made model ' + model_name + ' for subject ' + subject,
            'Accuracy': str(accuracy),
            'Loss': str(error)
        })

    elif model_name == 'MLP':
        model, accuracy, error = mlp(features, labels)
        dump(model, 'models.cla/' + subject + '-MLP.joblib')
        return jsonify({
            'Success': 'Made model ' + model_name + ' for subject ' + subject,
            'Accuracy': str(accuracy),
            'Loss': str(error)
        })

    else:
        return jsonify({
            'Error': 'Model not available'
        })
