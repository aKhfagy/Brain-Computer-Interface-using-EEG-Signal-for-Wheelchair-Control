import numpy as np
from flask import Flask, jsonify, request
from joblib import load
import traceback
from datasetreading import motor_imaginary
from APIHelperFunctions import choose_model_by_name

app = Flask(__name__)

MODELS = ['KNN', 'RF', 'MLP', 'SVM']
SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']


@app.route("/")
def home():
    return jsonify({
        'Predict': {
            'Models': MODELS,
            'Subjects': SUBJECTS,
            'Steps': ['1. Write subject name then a hyphen and then the model name to choose model. Ex: A-KNN',
                      '2. Write data path after that. Ex: test-data',
                      '3. Final path should look like Ex: /predict?model=A-KNN&data=data_featuresA']
        },
        'Make Models': {
            'Models': MODELS,
            'Subjects': SUBJECTS,
            'Steps': ['1. Write model name as \'model\' attribute',
                      '2. Write subject name as \'subject\' attribute',
                      '3. Final path should look like Ex: /make_models?model=KNN&subject=A']
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
        return jsonify({
            'Prediction': prediction
        })

    except:

        return jsonify({'trace': traceback.format_exc()})


@app.route('/make_models')
def make_models():
    subject = request.args.get('subject', default=None, type=str)
    model_name = request.args.get('model', default=None, type=str)

    return choose_model_by_name(model_name, subject)


@app.route('/preprocessing')
def preprocessing():
    motor_imaginary()

    return jsonify({
        'Finished preprocessing': 'Made features for CLA data',
        'Models': MODELS,
        'Subjects': SUBJECTS
    })


if __name__ == '__main__':
    app.run(debug=True)
