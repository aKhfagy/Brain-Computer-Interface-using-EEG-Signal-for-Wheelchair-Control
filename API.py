import numpy as np
from flask import Flask, jsonify, request
from joblib import load
import traceback

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({
        'Models': ['KNN', 'RF', 'MLP', 'SVM'],
        'Subjects': ['A', 'B', 'C', 'D', 'E', 'F'],
        'Steps': ['1. Write subject name then a hyphen and then the model name to choose model. Ex: A-KNN',
                  '2. Write data path after that. Ex: test-data',
                  '3. Final path should look like Ex: /predict?model=A-KNN&data=data_featuresA']
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


if __name__ == '__main__':
    app.run(debug=True)
