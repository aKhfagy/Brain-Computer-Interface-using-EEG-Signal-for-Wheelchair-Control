import traceback
import logging

import numpy as np
from flask import Flask, jsonify, request
from joblib import load

from APIHelperFunctions import choose_model_by_name, get_data, rmse
from datasetreading import motor_imaginary

app = Flask(__name__)

MODELS = ['KNN', 'RF', 'MLP', 'SVM']
SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']

logging.basicConfig(filename='predictions.log', level=logging.INFO)


@app.route("/")
def home():
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

    return jsonify({
        'Finished preprocessing': 'Made features for CLA data',
        'Models': MODELS,
        'Subjects': SUBJECTS
    })


@app.route('/view_model')
def view_model():
    subject = request.args.get('subject', default=None, type=str)
    model_name = request.args.get('model', default=None, type=str)

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


@app.route('/direct_chair')
def direct_chair():
    prediction = request.args.get('prediction', default=None, type=str)

    if prediction is None:
        return jsonify({
            'Error': 'Please enter a valid path with all the attributes'
        })


if __name__ == '__main__':
    app.run()
