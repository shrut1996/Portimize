from keras.models import load_model
import tensorflow as tf
import keras
import numpy as np
import sklearn.preprocessing as prep

def predict(portfolio_prices):
    portfolio_prices = portfolio_prices['Close']
    return portfolio_prices
    portfolio_prices.drop(['Volume', 'Close'], 1, inplace=True)
    portfolio_prices.astype(float)
    # portfolio_prices = portfolio_prices.dropna(axis=0, how='any')
    for i in portfolio_prices.columns:  # df.columns[w:] if you have w column of line description
        portfolio_prices[i] = portfolio_prices[i].fillna(portfolio_prices[i].median())
    portfolio_prices = normalize(portfolio_prices)
    X_train, y_train, X_test, y_test = preprocess_data(portfolio_prices[:: -1], 22)
    loaded_model = load_model('model.h5')
    adam = keras.optimizers.Adam(decay=0.2)
    loaded_model.compile(loss='mse',optimizer=adam, metrics=['accuracy'])
    # tf.keras.backend.clear_session()
    graph = tf.get_default_graph()
    with graph.as_default():
        return loaded_model.predict(X_train)


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