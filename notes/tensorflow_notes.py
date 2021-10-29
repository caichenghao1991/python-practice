''':key
    google colab similar to jupyter notebook

'''

import tensorflow as tf
from tensorflow import keras
import numpy as np

def example1():
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])   # sequential 1 layer 1 neuron
    model.compile(optimizer='sgd', loss='mean_squared_error')

    x = np.array([-1.0, 0, 1, 2, 3, 4], dtype=float)
    y = np.array([-3.0, -1, 1, 3, 5, 7], dtype=float)
    model.fit(x, y, epochs=500)
    print(model.predict([10]))
if __name__ == '__main__':
    example1()