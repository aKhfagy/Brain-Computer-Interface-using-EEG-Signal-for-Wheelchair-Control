# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 03:13:47 2022

@author: omnia
"""

import numpy as np
from sklearn.model_selection import train_test_split
def SVM(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, test_size=0.33,
                                                        random_state=42)
    from sklearn import svm

    clf = svm.SVC(kernel='linear') 
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    error = 0
    percentage = 0
    for i in range(len(predictions)):
        error = error + ((predictions[i] - y_test[i])**2)
        percentage = percentage + (1.0 if predictions[i] == y_test[i] else 0.0)

    percentage = percentage / len(predictions)
    error = np.sqrt(error)
    return clf, percentage, error