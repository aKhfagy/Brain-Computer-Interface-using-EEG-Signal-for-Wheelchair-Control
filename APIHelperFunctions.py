from KNN import knn
from MLP import mlp
from RandomForest import rf
from SVM import svm
from savedfiles import load_features_motor_dataset
from joblib import dump
from flask import jsonify
import numpy as np
import tensorflow as tf


def choose_model_by_name(model_name, subject):
    features, labels, n_outputs = load_features_motor_dataset('features.motor_dataset/data_features'
                                                              + subject + '.npy',
                                                              'features.motor_dataset/data_labels'
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


def get_data(subject):
    features, labels, n_outputs = load_features_motor_dataset('features.motor_dataset/data_features'
                                                              + subject + '.npy',
                                                              'features.motor_dataset/data_labels'
                                                              + subject + '.npy')
    return features, labels, n_outputs


def rmse(y_hat, y):
    error = 0
    percentage = 0
    for i in range(len(y_hat)):
        error = error + ((y_hat[i] - y[i])**2)
        percentage = percentage + (1.0 if y_hat[i] == y[i] else 0.0)

    percentage = percentage / len(y_hat)
    error = np.sqrt(error)
    return percentage, error


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(94, 94, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

