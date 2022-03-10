from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Reshape, LSTM, Dense, Lambda
import numpy as np


def make_model(Tx, n_a, n_values, reshapor, LSTM_cell, densor):
    X = Input(shape=(Tx, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0

    outputs = []
    for t in range(Tx):
        x = Lambda(lambda x: X[:, t, :])(X)
        x = reshapor(x)
        a, _, _ = LSTM_cell(x, [a, c])
        out = densor(a)
        outputs.append(out)

    model = Model(inputs=[X, a0, c0], outputs=outputs)
    return model


def compile_model(x, y, Tx, n_values, name):

    n_a = 64
    reshapor = Reshape((1, n_values))
    LSTM_cell = LSTM(n_a, return_state=True)
    densor = Dense(n_values, activation='softmax')

    model = make_model(Tx, n_a, n_values, reshapor, LSTM_cell, densor)

    model.summary()
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print('model.inputs')
    [print(i.shape, i.dtype) for i in model.inputs]
    print('model.outputs')
    [print(o.shape, o.dtype) for o in model.outputs]
    print('model.layers')
    [print(l.name, l.input_shape, l.dtype) for l in model.layers]
    m = n_values
    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))
    model.fit(np.array([x, a0, c0]), y, epochs=100)
    return model

