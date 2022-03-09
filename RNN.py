# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 19:59:23 2022

@author: omnia
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
from sklearn.model_selection import train_test_split


def RNN(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, test_size=0.33,
                                                        random_state=42)
    rnn = Sequential()
    print(X_train.shape)
    s= (X_train.shape)
    rnn.add(LSTM(units = 45, return_sequences = True, input_shape = s))
    rnn.add(Dropout(0.2))
    rnn.add(LSTM(units = 45, return_sequences = True))
    rnn.add(Dropout(0.2))
    rnn.add(LSTM(units = 45, return_sequences = True))
    rnn.add(Dropout(0.2))
    rnn.add(LSTM(units = 45))
    rnn.add(Dropout(0.2))
    for i in [True, True, False]:
      rnn.add(LSTM(units = 45, return_sequences = i))
      rnn.add(Dropout(0.2))

    rnn.add(Dense(units = 1))
    rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
    rnn.fit(X_train, y_train, epochs = 100, batch_size = 32)
    predictions = rnn.predict(X_test)
    error = 0
    percentage = 0
    for i in range(len(predictions)):
        error = error + ((predictions[i] - y_test[i])**2)
        percentage = percentage + (1.0 if predictions[i] == y_test[i] else 0.0)

    percentage = percentage / len(predictions)
    error = np.sqrt(error)
    return rnn, percentage, error













