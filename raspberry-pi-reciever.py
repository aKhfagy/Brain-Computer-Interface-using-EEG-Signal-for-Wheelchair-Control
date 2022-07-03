from flask import Flask, jsonify, request
import requests
import numpy as np
from joblib import load
import time
import MDD10A as HBridge


if __name__ == '__main__':
    url = "http://192.168.1.10:5000/get_chosen_model"

    try:
        response = requests.get(url)
        model = response.json()
        model_name = model['model']

        model = load('models.cla/' + model_name + '.joblib')
        data = np.load('features.motor_dataset/data_features' + model_name[0] + '.npy')
        predictions = model.predict(data)
        for prediction in predictions:
            url = "http://192.168.1.10:5000/direct_chair?prediction=" + str(prediction)
            requests.get(url)

            if prediction == 0:
                # right
                HBridge.setMotorLeft(1)
                HBridge.setMotorRight(-1)
            elif prediction == 1:
                # left
                HBridge.setMotorLeft(-1)
                HBridge.setMotorRight(1)
            elif prediction == 3:
                # forward
                HBridge.setMotorLeft(1)
                HBridge.setMotorRight(1)
            elif prediction == 5:
                # backwards
                HBridge.setMotorLeft(-1)
                HBridge.setMotorRight(-1)
            else:
                # stop
                HBridge.setMotorLeft(0)
                HBridge.setMotorRight(0)

            time.sleep(10)
    except requests.exceptions.ConnectionError:
        print('Connection Failed')
