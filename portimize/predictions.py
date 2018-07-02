from keras.models import load_model
import tensorflow as tf
import keras
import numpy as np
import sklearn.preprocessing as prep


def predict(portfolio_prices, period):
    portfolio_prices = portfolio_prices.iloc[:, 1:4].values.mean(axis=1)
    portfolio_prices = np.reshape(portfolio_prices, (-1, 1))
    sc = prep.MinMaxScaler()
    portfolio_prices = sc.fit_transform(portfolio_prices)
    portfolio_prices = np.reshape(portfolio_prices, (portfolio_prices.shape[0], 1, 1))
    if period==6:
        model_type='Models/5DayModel.h5'
    elif period==13:
        model_type = 'Models/10DayModel.h5'
    else:
        model_type = 'Models/15DayModel.h5'
    model = load_model(model_type)
    predicted_prices = model.predict(portfolio_prices)
    return sc.inverse_transform(predicted_prices)


def normalize(prices):
    min_max_scaler = prep.MinMaxScaler()
    prices['Open'] = min_max_scaler.fit_transform(prices.Open.values.reshape(-1, 1))
    prices['High'] = min_max_scaler.fit_transform(prices.High.values.reshape(-1, 1))
    prices['Low'] = min_max_scaler.fit_transform(prices.Low.values.reshape(-1, 1))
    prices['Adj Close'] = min_max_scaler.fit_transform(prices['Adj Close'].values.reshape(-1, 1))
    return prices


def standard_scaler(X_train, X_test):
    train_samples, train_nx, train_ny = X_train.shape
    test_samples, test_nx, test_ny = X_test.shape

    X_train = X_train.reshape((train_samples, train_nx * train_ny))
    X_test = X_test.reshape((test_samples, test_nx * test_ny))

    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    X_train = X_train.reshape((train_samples, train_nx, train_ny))
    X_test = X_test.reshape((test_samples, test_nx, test_ny))

    return X_train, X_test


def preprocess_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix()

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    row = result.shape[0]
    train = result[: int(row), :]

    train, result = standard_scaler(train, result)

    X_train = train[:, : -1]
    y_train = train[:, -1][:, -1]
    X_test = result[int(row):, : -1]
    y_test = result[int(row):, -1][:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return [X_train, y_train, X_test, y_test]