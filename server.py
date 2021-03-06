from statistics import mode
import traceback
import logging

import cv2
import numpy as np
from flask import Flask, jsonify, request
from joblib import load
import os as os
import werkzeug
from PIL import Image
import requests as requests
from io import BytesIO
import tensorflow as tf


from APIHelperFunctions import choose_model_by_name, get_data, rmse, create_model
from datasetreading import motor_imaginary

app = Flask(__name__)

logging.basicConfig(filename='predictions.log', level=logging.INFO)

terminating_string = '*!%)#!'

current_string = ''


@app.route("/")
def home():
    MODELS = ['KNN', 'RF', 'MLP', 'SVM']
    SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']

    return jsonify({
        'Models': MODELS,
        'Subjects': SUBJECTS,
        'Predict': {
            'Description': 'returns predictions for certain subject using the selected model on input data',
            'Steps': ['1. Write subject name then a hyphen and then the model name to choose model. Ex: A-KNN',
                      '2. Write data path after that. Ex: test-data',
                      '3. Final path should look like Ex: /predict?model=A-KNN&data=data_featuresA']
        },
        'Make Models': {
            'Description': 'Makes selected model for certain subject',
            'Steps': ['1. Write model name as \'model\' attribute',
                      '2. Write subject name as \'subject\' attribute',
                      '3. Final path should look like Ex: /make_models?model=KNN&subject=A']
        },
        'View Model': {
            'Description': 'Views selected model accuracy and loss on subject data (whole)',
            'Steps': ['1. Write model name as \'model\' attribute',
                      '2. Write subject name as \'subject\' attribute',
                      '3. Final path should look like Ex: /view_model?model=KNN&subject=A']
        },
        'Preprocessing': {
            'Description': 'Makes features for training',
            'Steps': 'path must be /preprocessing'
        }
    })


@app.route('/predict')
def predict():
    model = request.args.get('model', default=None, type=str)
    data = request.args.get('data', default=None, type=str)
    try:
        if model is None or data is None:
            return jsonify({
                'Error': 'Please enter params for data and model to proceed',
                'Example': '/predict?model=MODEL_NAME&data=DATA_NAME'
            })

        model = load('models.cla/' + model + '.joblib')

        data = np.load('features.motor_dataset/' + data + '.npy')

        prediction = model.predict(data)
        prediction = str(prediction.tolist())
        app.logger.info(prediction)

        return jsonify({
            'Prediction': prediction
        })

    except:

        return jsonify({'trace': traceback.format_exc()})


@app.route('/make_model')
def make_model():
    subject = request.args.get('subject', default=None, type=str)
    model_name = request.args.get('model', default=None, type=str)

    if subject is None or model_name is None:
        return jsonify({
            'Error': 'Please enter a valid path with all the attributes'
        })

    return choose_model_by_name(model_name, subject)


@app.route('/preprocessing')
def preprocessing():
    motor_imaginary()
    MODELS = ['KNN', 'RF', 'MLP', 'SVM']
    SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']

    return jsonify({
        'Finished preprocessing': 'Made features for CLA data',
        'Models': MODELS,
        'Subjects': SUBJECTS
    })


@app.route('/view_model')
def view_model():
    subject = request.args.get('subject', default=None, type=str)
    model_name = request.args.get('model', default=None, type=str)

    MODELS = ['KNN', 'RF', 'MLP', 'SVM']
    SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']

    if subject is None or model_name is None or subject not in SUBJECTS or model_name not in MODELS:
        return jsonify({
            'Error': 'Please enter a valid path with all the attributes'
        })

    x, y, _ = get_data(subject)

    model = load('models.cla/' + subject + '-' + model_name + '.joblib')

    y_hat = model.predict(x)

    percentage, loss = rmse(y_hat, y)

    return jsonify({
        'Accuracy': percentage,
        'Loss': loss
    })


@app.route('/predict_eye_movement', methods=['GET', 'POST'])
def predict_eye_movement():
    if request.method == 'POST':

        # labels:
        # close: 0, forward: 1, left: 2, right: 3

        if len(request.files) != 1:
            return jsonify({
                'ERROR': 'Wrong number of files'
            })
        file = request.files['pic']
        app.logger.info("Image recieved data type: " + str(type(file)))
        image_name = werkzeug.utils.secure_filename(file.filename)
        image_path = 'images.eyes/' + image_name
        file.save(image_path)
        file = cv2.imread(image_path)
        image = cv2.resize(file, (94, 94))
        norm = (image - np.mean(image)) / np.std(image)

        x = np.array(norm).reshape(-1, 94, 94, 3)
        x = x.astype('float32')

        model = create_model()
        model.load_weights('./models.eyetracker/eye_tracker')
        prediction = model.predict(x)

        return jsonify({
            'Result': str(prediction[0])
        })
    else:
        return jsonify({
            'Result': 'Please use post request'
        })


@app.route('/get_movement')
def get_movement():
    f = open('last_reading.txt', 'r')
    last_reading = int(f.readline())
    f.close()

    direction = 'stop'
    if last_reading == 0:
        direction = 'right'
    elif last_reading == 1:
        direction = 'left'
    elif last_reading == 3:
        direction = 'forwards'
    elif last_reading == 5:
        direction = 'backwards'

    f = open('counter.txt', 'r')
    counter = int(f.readline())
    f.close()

    if counter >= 3:
        logging.info('Displaying', direction, 'active')

        return jsonify({
            'Status': 'Active',
            'Direction': direction
        })

    logging.info('Displaying', direction, 'validating')
    return jsonify({
        'Status': 'Validating',
        'Direction': direction
    })


@app.route('/direct_chair')
def direct_chair():
    prediction = request.args.get('prediction', default=None, type=int)

    if prediction is None:
        return jsonify({
            'Error': 'Please enter a valid path with all the attributes'
        })
    else:
        prediction = int(prediction)

        f = open('last_reading.txt', 'r')
        last_reading = int(f.readline())
        f.close()

        f = open('counter.txt', 'r')
        counter = int(f.readline())
        f.close()

        if prediction == last_reading:
            counter += 1
        else:
            last_reading = prediction
            counter = 1

        f = open('last_reading.txt', 'w')
        f.write(str(last_reading))
        f.close()

        f = open('counter.txt', 'w')
        f.write(str(counter))
        f.close()

        if counter >= 3:
            logging.info('Moving to', str(prediction))

    return jsonify({
        'Prediction': str(prediction),
        'Counter': str(counter)
    })


@app.route('/chose_model')
def chose_model():
    subject = request.args.get('subject', default=None, type=str)
    model_name = request.args.get('model', default=None, type=str)

    if subject is None or model_name is None:
        return jsonify({
            'Response': 'Please try again'
        })

    f = open('model-cla.txt', 'w')
    f.write(subject + '-' + model_name + '.joblib')
    f.close()

    return jsonify({
        'Response': 'successful'
    })


@app.route('/get_chosen_model')
def get_chosen_model():
    f = open('model-cla.txt', 'r')
    model_name = f.readline()
    f.close()

    return jsonify({
        'model': model_name
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
