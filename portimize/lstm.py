import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import datetime
from sklearn import preprocessing
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

# # define a function to convert a vector of time series into a 2D matrix
# def convertSeriesToMatrix(vectorSeries, sequence_length):
#     matrix=[]
#     for i in range(len(vectorSeries)-sequence_length+1):
#         matrix.append(vectorSeries[i:i+sequence_length])
#     return matrix
#
# # random seed
# np.random.seed(1234)
#
# # load the data
# path_to_dataset = 'a.csv'
# sequence_length = 20
#
# path_to_test_dataset = 'b.csv'
#
# # vector to store the time series
# vector_vix = []
# with open(path_to_dataset) as f:
#     next(f) # skip the header row
#     next(f)
#     for line in f:
#         print line
#         fields = line.split(',')
#         print fields
#         vector_vix.append(float(fields[6]))
#
# vector_vix2 = []
# with open(path_to_test_dataset) as f:
#     next(f) # skip the header row
#     next(f)
#     for line in f:
#         print line
#         fields = line.split(',')
#         print fields
#         vector_vix2.append(float(fields[6]))
#
# # convert the vector to a 2D matrix
# matrix_vix = convertSeriesToMatrix(vector_vix, sequence_length)
# matrix_vix2 = convertSeriesToMatrix(vector_vix2, sequence_length)
#
# # shift all data by mean
# matrix_vix = np.array(matrix_vix)
# shifted_value = matrix_vix.mean()
# matrix_vix -= shifted_value
# print "Data  shape: ", matrix_vix.shape
#
# matrix_vix2 = np.array(matrix_vix2)
# shifted_value = matrix_vix2.mean()
# matrix_vix2 -= shifted_value
# print "Data  shape: ", matrix_vix2.shape
#
# # split dataset: 90% for training and 10% for testing
# train_row = int(round(0.9 * matrix_vix.shape[0]))
# train_set = matrix_vix[:, :]
#
# # shuffle the training set (but do not shuffle the test set)
# np.random.shuffle(train_set)
# # the training set
# X_train = train_set[:, :-1]
# # the last column is the true value to compute the mean-squared-error loss
# y_train = train_set[:, -1]
# # the test set
# X_test = matrix_vix2[:, :-1]
# y_test = matrix_vix2[:, -1]
#
# # the input to LSTM layer needs to have the shape of (number of samples, the dimension of each element)
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#
# # build the model
# model = Sequential()
# # layer 1: LSTM
# model.add(LSTM( input_dim=1, output_dim=50, return_sequences=True))
# model.add(Dropout(0.2))
# # layer 2: LSTM
# model.add(LSTM(output_dim=100, return_sequences=False))
# model.add(Dropout(0.2))
# # layer 3: dense
# # linear activation: a(x) = x
# model.add(Dense(output_dim=1, activation='linear'))
# # compile the model
# model.compile(loss="mse", optimizer="rmsprop")
#
# # train the model
# model.fit(X_train, y_train, batch_size=512, nb_epoch=50, validation_split=0.05, verbose=1)
#
# # evaluate the result
# test_mse = model.evaluate(X_test, y_test, verbose=1)
# print '\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(y_test))
#
# # get the predicted values
# predicted_values = model.predict(X_test)
# num_test_samples = len(predicted_values)
# predicted_values = np.reshape(predicted_values, (num_test_samples,1))
#
# # plot the results
# fig = plt.figure()
# plt.plot(y_test + shifted_value)
# plt.plot(predicted_values + shifted_value)
# plt.xlabel('Date')
# plt.ylabel('VIX')
# plt.show()
# fig.savefig('output_prediction.jpg', bbox_inches='tight')
#
# # save the result into txt file
# test_result = zip(predicted_values, y_test) + shifted_value
# np.savetxt('output_result.txt', test_result)

seq_len = 22
d = 0.2
shape = [4, seq_len, 1] # feature, window, output
neurons = [128, 128, 32, 1]
epochs = 40

def load_data(stock, seq_len):
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

df = pd.read_csv('../a.csv')
df.set_index('Date', inplace=True)
X_train, y_train, X_test, y_test = load_data(df, seq_len)


def build_model2(layers, neurons, d):
    model = Sequential()

    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))

    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(neurons[2], kernel_initializer="uniform", activation='relu'))
    model.add(Dense(neurons[3], kernel_initializer="uniform", activation='linear'))
    # model = load_model('my_LSTM_stock_model1000.h5')
    # adam = keras.optimizers.Adam(decay=0.2)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

model = build_model2(shape, neurons, d)
model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=epochs,
    validation_split=0.1,
    verbose=1)

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
print trainScore[0]
print testScore[0]

def percentage_difference(model, X_test, y_test):
    percentage_diff=[]

    p = model.predict(X_test)
    for u in range(len(y_test)): # for each data index in test data
        pr = p[u][0] # pr = prediction on day u

        percentage_diff.append((pr-y_test[u]/pr)*100)
    return p

p = percentage_difference(model, X_test, y_test)

import pandas_datareader.data as web

def denormalize(stock_name, normalized_value):
    start = datetime.datetime(2000, 1, 1)
    end = datetime.date.today()
    df = web.DataReader(stock_name, "yahoo", start, end)

    df = df['Adj Close'].values.reshape(-1, 1)
    normalized_value = normalized_value.reshape(-1, 1)

    # return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new

import matplotlib.pyplot as plt2

def plot_result(stock_name, normalized_value_p, normalized_value_y_test):
    newp = denormalize(stock_name, normalized_value_p)
    newy_test = denormalize(stock_name, normalized_value_y_test)
    plt2.plot(newp, color='red', label='Prediction')
    plt2.plot(newy_test,color='blue', label='Actual')
    plt2.legend(loc='best')
    plt2.title('The test result for {}'.format(stock_name))
    plt2.xlabel('Days')
    plt2.ylabel('Adjusted Close')
    plt2.show()
    return newy_test

new = plot_result('^GSPC', p, y_test)
model.save('model.h5')

# model = load_model('my_LSTM_stock_model1000.h5')
# adam = keras.optimizers.Adam(decay=0.2)
# model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
# model.summary()
# return model




