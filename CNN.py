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


def CNN(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, test_size=0.33,
                                                        random_state=42)
    final = len(X_train) / 2
    final = int(final)
    no_feature = 4

    u = X_train[0:final]
    print(u.shape)
    u = np.reshape(u, (1, final, 4, 1))
    v = X_train[final:]
    print(v.shape)
    v = np.reshape(v, (1, final + 1, 4, 1))

    uy = y_train[0:final]
    vy = y_train[final:]

    inp1 = Input(shape=(final, no_feature), dtype='float32', name='c1input')
    conv1 = Conv1D(128, kernel_size=20, strides=10, input_shape=(final, no_feature), name='audConv_l1')(inp1)
    conv1 = Flatten(name='audConv_l2')(conv1)
    conv1 = Dropout(0.3, name='audConv_l3')(conv1)
    a1 = Dense(256, activation='relu', name='rate_l1')(conv1)
    a1 = Dropout(0.15, name='rate_l2')(a1)
    inp2 = Input(shape=(final, no_feature), dtype='float32', name='c2input')
    conv2 = Conv1D(128, kernel_size=20, strides=10, input_shape=(final + 1, no_feature), name='audConv_21')(inp2)
    conv2 = Flatten(name='audConv_22')(conv2)
    conv2 = Dropout(0.3, name='audConv_23')(conv2)
    a2 = Dense(256, activation='relu', name='rate_21')(conv2)
    a2 = Dropout(0.15, name='rate_22')(a2)
    AV = concatenate([a1, a2], name='AVRate_l1')
    decOutput = Dense(40, activation='softmax', name='decOutput')(AV)
    model = Model(inputs=[inp1, inp2], outputs=decOutput)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # print(model.summary())

    X = X.astype('float32')
    y1 = np_utils.to_categorical(y_train, 5)
    model.fit([u, v], y_train, epochs=5)
    model.evaluate(X_test, y_test)
    return model
