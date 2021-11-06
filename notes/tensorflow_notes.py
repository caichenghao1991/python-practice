'''
    google colab similar to jupyter notebook
    Scikit-learn
        mainly for machine learning, not for deep learning, don't support GPU
    Caffe
        first deep learning framework, no auto-grad
    Keras
        wrapper, no implementation
    Theano
        hard develop and debug
    Torch, Pytorch
        better for researching, not production. Pytorch has Torch frontend and Caffe backend

    Eco system
        TensorFlow.js (javascript)   Tensorflow Lite (mobile)  Tensorflow Extended (Production)
        TPU(Tensor processing unit) cloud accelerate (cheap and fast)


    import tensorflow as tf
    a = tf.constant(1.)
    b = tf.constant(2.)
    c = tf.add(a,b)
    print(float(c))

    gpu accelerate
        def gpu_run():
            with tf.device('/gpu:0'):
                gpu_a = tf.random.normal([10000,1000])
                gpu_b = tf.random.normal([1000,2000])
                c = tf.matmul(gpu_a,gpu_b)
        gpu_time = timeit.timeit(gpu_run, number=10)  # cpu time 0.611148, gpu time 0.0008749000000003448

    auto gradient
        x, a, b, c = tf.constant(1.0), tf.constant(2.0), tf.constant(3.0), tf.constant(4.0)  # must be float
        with tf.GradientTape() as tape:
            tape.watch([a, b, c])
            y = a ** 2 * x + b * x + c
        [dy_da, dy_db, dy_dc] = tape.gradient(y, [a, b, c])
        print(dy_da, dy_db, dy_dc)

    common neural network api
        tf.matmul,tf.nn.conv2d, tf.nn.relu, tf.nn.max_pool2d,  tf.nn.sigmoid,  tf.nn.softmax
        layers.Dense, layers.Conv2D, layers.SimpleRNN, layers.LSTM, layers.ReLU, layers.MaxPool2D

    installation
        cuda:  custom install, skip NVIDIA Geforce Experience, under options: cuda: skip Visual Studio Integration if
            visual studio not installed, driver components, skip if current version higher than new version
        cudnn: make sure version suitable for cuda version, copy all files to cuda installed same name folder
        system config: path add bin, libnvvp. tifether and move to top
        terminal test: nvcc -V

        pip install tensorflow
        where ipython      ipython
        import tensorflow as tf   tf.test.is_gpu_available()


    Device
        has cpu and gpu device, operation must be done on same device
        with tf.device('gpu'):  # 'cpu'
            a = tf.constant(1)
        print(a.device)  # return device (CPU/GPU)
        aa = a.cpu()   # convert to cpu
        aaa = aa.gpu()  # convert to gpu


    Data type
        int, float, double, bool, string
        tf.constant(1)  # int32   # return created tensor (id, shape, dtype, numpy)
                        # tf.Tensor(1, shape=(), dtype=int32)
            # tf.constant(1.)  # float32
            # tf.constant(2.2, dtype=tf.int32)  # raise error
            # tf.constant(2., dtype=tf.double)   # float64
            # tf.constant([True, False])    # bool
            # tf.constant('Hello world.')

    Properties
        a.numpy()      # return numpy array
        a.ndim          # return 0 for scalar
        tf.rank(a)      # return Tensor with numpy value equal to rank (ndim)
        a.shape         # return shape of tensor
        a.dtype         # data type

        tf.is_tensor(a)   # check whether object is tensor

        tf.cast(a2, dtype=tf.float32)                 # cast dtype to specified type
        tf.cast(a2, dtype=tf.bool)                    # true 1, false 0
        int(a), float(a)   # return 1, work for only scalar

    Create tensor
        a2 = tf.convert_to_tensor(arr, dtpe=int32)    # convert numpy array or list to tensor, dtype keep original if
                                                      # not specified
        a = tf.constant(1.)  # float32

        a = tf.range(5)
        a = tf.ones([])     # a = 1,  shape inside param  shape ()
        a = tf.zeros([1])      # a = [0]  tensor     shape (1,)
        b = tf.zeros_like(a)    # create tensor with same shape of a, filled with 0
            # b = tf.ones_like(a)
        a = tf.fill([2,2], 3)    # create a 2*2 tensor filled with 3

        a = tf.random.normal([2,2],mean=1, stddev=1)    # 2*2 tensor with normal distribution, default mean 0, std 1
        a = tf.random.truncated_normal([2,2],mean=1, stddev=1)   # resample if outside [-2*std+mean, 2*std+mean]
        a = tf.random.uniform([2,2], minval=1, maxval=1)   # 2*2 tensor with uniform distribution, dtype optional
        index = tf.random.shuffle(ind)


    Variable
        a = tf.range(5)
        a = tf.Variable(a)  # variable is able to take gradient (usually for learned weights and biased)
        a.trainable         # check tensor whether able to take gradient


        loss = tf.keras.losses.mse(y, out)   # mean squared error
        avg = tf.reduce_sum(a) / a.shape[0]  # same as tf.reduce_mean(a)

    tensor dimension:
        2D:  batch size, column size
        3D: batcg size, # words, embedding size
        4D: batch size, height, width, color

    kernel: wight w trainable variable
    bias: b trainable variable

    indexing
    a = tf.range(10)
    a[-2:]  #[8,9]
    a[1:6]  #[1,2,3,4,5]  # not inclusive right side
    a[1:8:2]  # [1,3,5,7]  # step 2
    a[-1::-3]  # [9,6,3]  same as [::-3]
    a[6:0:-3]  # [6,3]

    a = tf.ones([4,28,28,3])
    a[0][0][0]   # tensor [1, 1, 1]  # index from outer to inner
    a[0, 0, 0]   # same as a[0][0][0]    # shape(28,28,3)
    a[0,:,:,:]  # same as a[0]  # : use all index
    a[::2, :, :, :]  # same as a[0] and a[2]   start:end:step  start default 0, end default -1
    a[0,...,1,:]   # ... can be any length of :, must able to infer, can't use 2 ... same time    # shape[28,3]

    a = tf.gather(a, axis=0, indices =[1,2])  # default axis=0  return an ordered tensor with selected indices in axis,
            # shape(2,28,28,3), axis=0 means in pick in first dimension (4) inside a (4,28,28,3)

    tf.gather_nd(a,[0,1,2])      # gather the item with indices of several dimension, shape (3), get the item with index
                                 # 0 in 1st dimension, 1 in second dimension, 2 in third dimension
        tf.gather_nd(a,[[0,1,2]])   # shape (1,3)
        tf.gather_nd(a,[[0,1,2],[0,1,3]])   # get 2 items as tensors with indices, shape (2,3)
    tf.boolean_mask(a, mask=[True, True, False, False])   # use boolean mask to pick items needed, here is first 2 items
                                                        # in first dimension, shape (2,28,28,3), mask array length need
                                                        # same as axis length, default axis = 0
    tf.boolean_mask(tf.ones([2,3,4]), mask=[[True, True, False],[False,False,True]])  # 2D mask same shape as first 2D
                                                        # of target tensor (2,3)

    a = tf.ones([4,28,28,3])
    b = tf.reshape([4,784,3])
    c = tf.reshape([4,-1,3])   # shape (4,784,3), only allow one -1, automatically calculate size

    a = tf.ones([4,20,28,3])
    b = tf.transpose(a)   # shape (3,28,20,4)
    c = tf.transpose(a,[0,3,1,2])   # shape (4,3,20,28)

    b = tf.expend_dims(a, axis=3)   # shape (4,20,28,1,3)
    c = tf.squeeze(tf.zeros([1,2,1,1,3]))     # shape (2,3)
    c = tf.squeeze(tf.zeros([1,2,1,3]), axos=0)     # shape (2,1,3), only able to squeeze dimension size = 1

    broadcast
        default align right side, if small dimension tensor size 1, can be extended to the other tensor size, if both
            has different size, it's not able to broadcast. broadcast won't change tensor
        bias shape (1,28,1) can broadcast when add a (4,28,28,3). it will copy to size (4,28,28,3)
        default small dimension add to large dimension, suitable for all entry
        a shape(4,1) add b shape(1,3)  result (4,3) with broadcast, while (1,4) and (1,3) can't be added together
        a = tf.ones([4,20,28,3])
        print(a + tf.random.normal([28,1])).shape   #    implicit broadcast to shape (4,20,28,3)
        b = tf.broadcast_to(tf.random.normal([28,1]),[4,20,28,3])   # explicit declare

        b = tg.expand_dims(tf.random.normal([28,1]), axis=0)
        b = tg.expand_dims(b, axis=0)  # shape (1,1,28,1)
        b2 = tf.tile(b,[4,20,1,3])   # create tensor with shape (4,20,28,3)  repeating b, same functionality as
                                     # broadcast, but use more memory

    math operations
        +  -  *  /     **  pow  square      sqrt     //  %     tf.exp  tf.math.log (e base)     element-wise
        @  matmul (row i in matrix a times column j in matrix b, get result index [i,j] )     matrix-wise
            matrix a shape (4,2,3), matmul matrix b shape (4,3,5)   # return [4,2,5], 4 is batch size
        reduce_mean  max  min  sum   axis-wise

'''
import os

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers
from scipy import misc
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import timeit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def speed():
    print(tf.config.list_physical_devices('GPU'))
    with tf.device('/cpu:0'):
        cpu_a = tf.random.normal([10000,1000])
        cpu_b = tf.random.normal([1000,2000])
        print(cpu_a.device)

    with tf.device('/gpu:0'):
        gpu_a = tf.random.normal([10000,1000])
        gpu_b = tf.random.normal([1000,2000])
        print(gpu_a.device)

    def cpu_run():
        with tf.device('/cpu:0'):
            c = tf.matmul(cpu_a,cpu_b)
        return c

    def gpu_run():
        with tf.device('/gpu:0'):
            c = tf.matmul(gpu_a,gpu_b)
        return c

    # warm up
    cpu_time = timeit.timeit(cpu_run, number=10)
    gpu_time = timeit.timeit(gpu_run, number=10)
    print("warmup", cpu_time, gpu_time)  # warmup 0.6254491 0.5488910000000002

    cpu_time = timeit.timeit(cpu_run, number=10)
    gpu_time = timeit.timeit(gpu_run, number=10)
    print("run time", cpu_time, gpu_time)  # run time 0.5999606000000002 0.0008114999999992989

    x, a, b, c = tf.constant(1.0), tf.constant(2.0), tf.constant(3.0), tf.constant(4.0)  # must be float
    print(a.numpy())
    with tf.GradientTape() as tape:
        tape.watch([a, b, c])
        y = a ** 2 * x + b * x + c
    [dy_da, dy_db, dy_dc] = tape.gradient(y, [a, b, c])
    print(dy_da, dy_db, dy_dc)


def mnist_digits():
    (x, y), (x_val, y_val) = datasets.mnist.load_data()
    x = tf.convert_to_tensor(x, dtype=tf.float32)/255
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)   # one hot encoding with 10 columns, with label class column 1, 9 columns 0
    print(x.shape, y.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
    train_dataset =train_dataset.batch(128)   # default 1 image a step, here batch 128 images

    model = keras.Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10)
    ])
    optimizer = optimizers.SGD(learning_rate=0.001)

    def train_epoch(epoch):
        for step, (x, y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                x = tf.reshape(x, (-1,28*28))    # [b,28,28] => [b, 784]
                out = model(x)   # output  [1,10]
                loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]  # loss function

            grads = tape.gradient(loss, model.trainable_variables)  # w, b in each layers
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss: ', loss.numpy())
    def train():
        for epoch in range(30):
            train_epoch(epoch)

    train()


def test():
    a = tf.ones([4, 28, 5, 3])
    b=tf.ones([1, 5,1])
    print(a+b)

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

def mnist_fashion():
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


def convolution_pooling():
    img = misc.ascent()
    plt.gray()   # change the cmap to gray, do not convert color image to gray scale
    #plt.axis('off')
    plt.grid(False)
    plt.imshow(img)
    #plt.show()

    img_tran = np.copy(np.array(img))
    height, width = img_tran.shape[0], img_tran.shape[1]
    weight = 1
    filter = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])  # activates horizontal lines
    #filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # activates straight lines
    # apply filter to do convolution, sum of (multiply same location value of filter and image) + bias
    for i in range(1, height-1):
        for j in range(1, width-1):
            box = img[i - 1:i + 2, j - 1:j + 2]
            box[1,0] = img[i,j-1]
            res = np.sum(box * filter) + weight
            res = min(max(0, res), 255)  # clip 0, 255
            img_tran[i, j] = res
            '''
            res = 0.0
            res = res +(img[x-1, y-1]*filter[0][0])
            res = res + (img[x, y - 1] * filter[0][1])
            res = res + (img[x + 1, y - 1] * filter[0][2])
            res = res + (img[x - 1, y] * filter[1][0])
            res = res + (img[x, y] * filter[1][1])
            res = res + (img[x + 1, y] * filter[1][2])
            res = res + (img[x - 1, y + 1] * filter[2][0])
            res = res + (img[x, y + 1] * filter[2][1])
            res = res + (img[x + 1, y + 1] * filter[2][2])
            res = res + weight
            if res<0:
                res=0
            if res>255:
                res=255

            img_tran[x, y] = res
            '''
    plt.imshow(img_tran)
    plt.show()

    # max pooling
    height_p, width_p = img_tran.shape[0]//2, img_tran.shape[1]//2
    img_pool = np.zeros((height_p, width_p))
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            pix = np.max(img_tran[i:i + 2, j:j + 2])
            img_pool[i//2,j//2] = pix
    plt.gray()
    plt.imshow(img_pool)
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
    #mnist_fashion()
    #convolution_pooling()
    #speed()
    #mnist_digits()
    test()