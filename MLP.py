from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np


def mlp(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    classifier = MLPClassifier(hidden_layer_sizes=(128, ), max_iter=1000, activation='relu', solver='adam')
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    error = 0
    percentage = 0
    for i in range(len(predictions)):
        error = error + ((predictions[i] - y_test[i])**2)
        percentage = percentage + (1.0 if predictions[i] == y_test[i] else 0.0)

    percentage = percentage / len(predictions)
    error = np.sqrt(error)
    return classifier, percentage, error
    