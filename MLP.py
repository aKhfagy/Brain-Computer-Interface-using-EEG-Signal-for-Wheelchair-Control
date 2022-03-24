# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 02:47:04 2022

@author: omnia
"""

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
def MLP(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, test_size=0.33,
                                                        random_state=42)
    classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    error = 0
    percentage = 0
    for i in range(len(predictions)):
        error = error + ((predictions[i] - y_test[i])**2)
        percentage = percentage + (1.0 if predictions[i] == y_test[i] else 0.0)

    percentage = percentage / len(predictions)
    error = np.sqrt(error)
    return classifier, percentage, error
    