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

    '''
    X_train, X_test = X_train / 255.0, X_test / 255.0
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),  # 128 nodes in the hidden layer
        keras.layers.Dense(10, activation=tf.nn.softmax)  # 10 class
    ])
    '''
    X_train, X_test = X_train.reshape(60000, 28, 28, 1), X_test.reshape(10000, 28, 28, 1)
    X_train, X_test = X_train / 255.0, X_test / 255.0
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),  # filters count, filter shape
        keras.layers.MaxPool2D(2,2),  # max pooling
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),  # 128 nodes in the hidden layer
        keras.layers.Dense(10, activation='softmax')  # 10 class
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()
    model.fit(X_train, y_train, epochs=5 ) #, callbacks=[callbacks]
    y_pred = model.predict(X_test)  # return probability array for each class

    print(model.evaluate(X_train, y_train))

    m = tf.keras.metrics.Accuracy()
    m.update_state(y_test, np.argmax(y_pred, axis=1))

    print(m.result().numpy()) # accuracy 0.887   0.913

    f, axes = plt.subplots(3, 4)
    FIRST_IMAGE = 4
    SECOND_IMAGE = 7
    THIRD_IMAGE = 26
    CONVOLUTION_NUMBER = 4
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    for x in activation_model.predict(X_test[FIRST_IMAGE].reshape(1,28,28,1)):
        print(x.shape)

    for x in range(0, 4):
        f1 = activation_model.predict(X_test[FIRST_IMAGE].reshape(1,28,28,1))[x]

        axes[0,x].imshow(f1[0,:,:, CONVOLUTION_NUMBER],cmap='inferno')
        axes[0,x].grid(False)
        f2 = activation_model.predict(X_test[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
        axes[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axes[1, x].grid(False)
        f3 = activation_model.predict(X_test[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
        axes[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axes[2, x].grid(False)
    plt.show()

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