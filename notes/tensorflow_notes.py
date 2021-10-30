''':key
    google colab similar to jupyter notebook

'''
import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class my_call_back(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss')<0.2:
            print("\nLoss is good enough, cancelling training")
            self.model.stop_training = True


def example1():
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])   # sequential 1 layer 1 neuron
    model.compile(optimizer='sgd', loss='mean_squared_error')

    x = np.array([-1.0, 0, 1, 2, 3, 4], dtype=float)
    y = np.array([-3.0, -1, 1, 3, 5, 7], dtype=float)
    model.fit(x, y, epochs=500)
    print(model.predict([10]))

def mnist():
    callbacks = my_call_back()
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train, y_train),(X_test, y_test) = fashion_mnist.load_data()   # 28*28

    X_train, X_test = X_train / 255.0, X_test / 255.0

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),  # 128 nodes in the hidden layer
        keras.layers.Dense(10, activation=tf.nn.softmax)  # 10 class
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(X_train, y_train, epochs=20, callbacks=[callbacks])
    y_pred = model.predict(X_test)  # return probability array for each class
    print(type(y_pred))
    print(model.evaluate(X_train, y_train))

    print(y_test.shape, y_pred.shape)
    m = tf.keras.metrics.Accuracy()
    m.update_state(y_test, np.argmax(y_pred, axis=1))

    print(m.result().numpy()) # accuracy 0.43


def cifar10():
    callback = my_call_back()
    X_train, X_test, y_train, y_test = [], [], [], []
    count=0
    for category in os.listdir('../resources/data/cifar10/train'):
        for name in os.listdir(os.path.join('../resources/data/cifar10/train', category)):
            temp = plt.imread(os.path.join('../resources/data/cifar10/train/', category, name))
            #temp = (0.299 * temp[:, :, 0] + 0.587 * temp[:, :, 1] + 0.114 * temp[:, :, 2]) / 3
            X_train.append(np.array(temp))
            y_train.append(count)
        count += 1
    count = 0
    for category in os.listdir('../resources/data/cifar10/test'):
        for name in os.listdir(os.path.join('../resources/data/cifar10/test', category)):
            temp = plt.imread(os.path.join('../resources/data/cifar10/test/', category, name))
            #temp = (0.299 * temp[:, :, 0] + 0.587 * temp[:, :, 1] + 0.114 * temp[:, :, 2]) / 3
            X_test.append(temp)
            y_test.append(count)
        count += 1
    #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


if __name__ == '__main__':
    #example1()
    mnist()