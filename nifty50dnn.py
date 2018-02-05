import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

np.random.seed(1)
data = pd.read_csv('a.csv')
data.dropna(axis=0, how='any')
data = data.drop(['Date'],axis=1)
arr = data.copy()
arr = arr.dropna(axis=0, how='any')
train_start=0
train_end=int(np.floor(0.8*arr.shape[0]))
test_start=train_end+1
test_end=int(arr.shape[0])
arr = arr.values
# shuffle_indices = np.random.permutation(np.arange(2466))
# arr=arr[shuffle_indices]
data_train=arr[np.arange(train_start, train_end),:]
data_test=arr[np.arange(test_start,test_end),:]
data_train=pd.DataFrame(data_train)
data_test=pd.DataFrame(data_test)
# for i in range(0,5):
#     data_train=data_train.loc[data_train[i]!='null',:]
#     data_test=data_test.loc[data_test[i]!='null',:]
# data_train=data_train.astype(float)
print data_train
# data_test=data_test.astype(float)
# scaler=MinMaxScaler()
# scaler.fit(data_train)
# data_train=scaler.transform(data_train)
# data_test=scaler.transform(data_test)
x_train=data_train.iloc[:,1:]
y_train=data_train.iloc[:, 0]
x_test=data_test.iloc[:,1:]
y_test=data_test.iloc[:, 0]
features = 4
X = tf.placeholder(dtype=tf.float32, shape=[None, features])
Y = tf.placeholder(dtype=tf.float32, shape=[None])
n_neurons_1 = 20
n_neurons_2 = 10
n_neurons_3 = 5
n_target = 1
sigma = 1
weight_initializer = keras.initializers.VarianceScaling(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.constant_initializer()
W_hidden_1 = tf.Variable(weight_initializer([features, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
W_out = tf.Variable(weight_initializer([n_neurons_3, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
out = (tf.add(tf.matmul(hidden_3, W_out), bias_out))
# mae=tf.reduce_mean(tf.abs(tf.subtract(Y,out)))
rmse=tf.sqrt(tf.reduce_mean(tf.squared_difference(out, Y)))
# mse = tf.reduce_mean(tf.squared_difference(out, Y))
opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=.99, beta2=.999).minimize(rmse)
net = tf.Session()
net.run(tf.global_variables_initializer())
# Number of epochs and batch size
epochs = 100
batch_size = 264

for e in range(epochs):

    # Shuffle training data
    # shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    # x_train = x_train.iloc[shuffle_indices,:]
    # y_train = y_train.iloc[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = x_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        # if np.mod(i, 5) == 0:
            # Prediction
            # pred = net.run(out, feed_dict={X: x_test})
            # line2, = ax1.plot(pred)
            # plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            # file_name = 'epoch_' + str(e) + '_batch_' + str(i)
            # plt.savefig(file_name+'.jpg')
            # plt.pause(0.01)

pred=net.run(out, feed_dict={X:x_test})
print pred
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test, linewidth=0.5)
line2, = ax1.plot(pred, linewidth=0.5)
# plt.savefig('nn3.jpeg')
#testing on very old 2003 data (out of range)
a=[ 1699.70 ,1728.00 ,1699.70, 1723.95 ]
pred2=net.run(out, feed_dict={X:pd.DataFrame(a).T})
print(pred2)
#printing mae
print(np.sum(np.abs(np.subtract(pred,(y_test.values.reshape(len(pred),1)))))/len(pred))
