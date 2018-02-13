import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
import pandas as pd
import math
import datetime
import pandas_datareader.data as web
from sklearn import preprocessing
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

seq_len = 22
d = 0.2
shape = [4, seq_len, 1] # feature, window, output
neurons = [128, 128, 32, 1]
epochs = 40


def preprocess(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix()
    sequence_length = seq_len + 1  # index starting from 0
    result = []

    for index in range(len(data) - sequence_length):  # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length])  # index : index + 22days

    result = np.array(result)
    row = round(0.9 * result.shape[0])  # 90% split

    train = result[:int(row), :]  # 90% date
    X_train = train[:, :-1]  # all data until day m
    y_train = train[:, -1][:, -1]  # day m + 1 adjusted close price

    X_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return [X_train, y_train, X_test, y_test]

# df = pd.read_csv('../a.csv')
# df.set_index('Date', inplace=True)
# X_train, y_train, X_test, y_test = preprocess(df, seq_len)
#
# model = Sequential()
#
# model.add(LSTM(neurons[0], input_shape=(shape[1], shape[0]), return_sequences=True))
# model.add(Dropout(d))
#
# model.add(LSTM(neurons[1], input_shape=(shape[1], shape[0]), return_sequences=False))
# model.add(Dropout(d))
#
# model.add(Dense(neurons[2], kernel_initializer="uniform", activation='relu'))
# model.add(Dense(neurons[3], kernel_initializer="uniform", activation='linear'))
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.summary()
#
# model.fit(
#     X_train,
#     y_train,
#     batch_size=512,
#     epochs=epochs,
#     validation_split=0.1,
#     verbose=1)
#
# p = percentage_difference(model, X_test, y_test)
#
# trainScore = model.evaluate(X_train, y_train, verbose=0)
# print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
#
# testScore = model.evaluate(X_test, y_test, verbose=0)
# print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
# print trainScore[0]
# print testScore[0]
#
#
# model.save('model.h5')