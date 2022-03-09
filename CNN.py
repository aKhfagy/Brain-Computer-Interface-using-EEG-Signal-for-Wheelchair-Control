# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:53:58 2022

@author: omnia
"""
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Embedding, concatenate
from keras.layers import Conv1D
from keras import optimizers
from keras.utils import np_utils
import numpy as np


def CNN(X, y, n_outputs):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, test_size=0.20,
                                                        random_state=42)
    final = len(X_train)
    final = int(final)
    no_col = 200
    print(X_train.shape)
    u = np.zeros((1, final, no_col))
    v = np.zeros((1, final, no_col))
    print(u.shape)
    print(v.shape)
    for i in range(len(X_train) - 1):
        for j in range(len(X_train[i][0])):
            u[0][i][j] = X_train[i][0][j]
        for j in range(len(X_train[i][1])):
            u[0][i][j] = X_train[i][1][j]

    inp1 = Input(shape=(final, no_col), dtype='float32', name='c1input')
    conv1 = Conv1D(128, kernel_size=20, strides=10, input_shape=(final, no_col), name='audConv_l1')(inp1)
    conv1 = Flatten(name='audConv_l2')(conv1)
    conv1 = Dropout(0.3, name='audConv_l3')(conv1)
    a1 = Dense(256, activation='relu', name='rate_l1')(conv1)
    a1 = Dropout(0.15, name='rate_l2')(a1)
    inp2 = Input(shape=(final, no_col), dtype='float32', name='c2input')
    conv2 = Conv1D(128, kernel_size=20, strides=10, input_shape=(final, no_col), name='audConv_21')(inp2)
    conv2 = Flatten(name='audConv_22')(conv2)
    conv2 = Dropout(0.3, name='audConv_23')(conv2)
    a2 = Dense(256, activation='relu', name='rate_21')(conv2)
    a2 = Dropout(0.15, name='rate_22')(a2)
    AV = concatenate([a1, a2], name='AVRate_l1')
    decOutput = Dense(40, activation='softmax', name='decOutput')(AV)
    model = Model(inputs=[inp1, inp2], outputs=decOutput)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # print(model.summary())

    y_temp = []
    for y in y_train:
        y_temp.append(y)
    for y in y_train:
        y_temp.append(y)
    y_train = y_temp
    y_train = np_utils.to_categorical(y_train, n_outputs)

    model.fit([u, v], y_train, epochs=100)
    u = []
    v = []
    for reading in X_test:
        u.append(reading[0])
        v.append(reading[1])
    print(u.shape)
    print(v.shape)
    #model.evaluate(X_test, y_test)
    return model
