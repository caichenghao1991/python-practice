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
        tf.matmul,tf.nn.conv2d, tf.nn.relu, tf.nn.max_pool2d,  tf.nn.sigmoid,  tf.nn.softmax, tf.nn.leaky_relu
        layers.Dense, layers.Conv2D, layers.SimpleRNN, layers.LSTM, layers.ReLU, layers.MaxPool2D
        layers.BatchNormalization()   layers.Activation('relu')  layers.UpSampling2D(size=3)  layers.Dropout(0.4)
        CONV/FC -> ReLu(or other activation) -> BatchNorm  -> Pooling -> Dropout

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
        int, float, double(more decimal place, double space), bool, string
        tf.constant(1)  # int32   # return created tensor (id, shape, dtype, numpy)
                        # tf.Tensor(1, shape=(), dtype=int32)
            # tf.constant(1.)  # float32
            # tf.constant(2.2, dtype=tf.int32)  # raise error
            # tf.constant(2., dtype=tf.double)   # float64
            # tf.constant([True, False])    # bool
            # tf.constant('Hello world.')
        tf.cast(a, dtype=tf.int32)  # casting to dtype

    Properties
        a.numpy()      # return numpy array
        a.ndim          # return 0 for scalar
        tf.rank(a)      # return Tensor with numpy value equal to rank (ndim)
        a.shape         # return shape of tensor
        a.dtype         # data type

        tf.is_tensor(a)   # check whether object is tensor

        tf.cast(a2, dtype=tf.float32)                 # cast dtype to specified type, a2 can be numpy array or tensor
        tf.cast(a2, dtype=tf.bool)                    # true 1, false 0
        int(a), float(a)   # return 1, work for only scalar

    Create tensor
        a2 = tf.convert_to_tensor(arr, dtype=int32)    # convert numpy array or list to tensor, dtype keep original if
                                                      # not specified
        a = tf.constant(1.)  # float32

        a = tf.range(5)
        a = tf.ones([])     # a = 1,  shape inside param  shape ()
        a = tf.zeros((1,))      # a = [0]  tensor     shape (1,)
        b = tf.zeros_like(a)    # create tensor with same shape of a, filled with 0
            # b = tf.ones_like(a)
        a = tf.fill((2,2), 3)    # create a 2*2 tensor filled with 3

        a = tf.random.normal((2,2),mean=1, stddev=1)    # 2*2 tensor with normal distribution, default mean 0, std 1
        a = tf.random.truncated_normal((2,2),mean=1, stddev=1)   # resample if outside [-2*std+mean, 2*std+mean]
        a = tf.random.uniform((2,2), minval=1, maxval=1)   # 2*2 tensor with uniform distribution, dtype optional
        index = tf.random.shuffle(ind)
        tf.random.set_seed(8)

    Variable
        a = tf.range(5)
        a = tf.Variable(a)  # variable is able to take gradient (usually for learned weights and biased)
        a.trainable         # check tensor whether able to take gradient


        loss = tf.keras.losses.mse(y, out)   # mean squared error
        avg = tf.reduce_sum(a) / a.shape[0]  # same as tf.reduce_mean(a)

    tensor dimension:
        2D:  batch size, column size
        3D: batch size, # words, embedding size
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
        a[0, 0, 0]   # same as a[0][0][0]    # shape(3)
        a[0,:,:,:]  # same as a[0]  # : use all index
        a[::2, :, :, :]  # same as a[0] and a[2]   start:end:step  start default 0, end default -1
        a[0,...,1,:]   # ... can be any length of :, must able to infer, can't use 2 ... same time    # shape[28,3]

    gather
        a = tf.gather(a, axis=0, indices =[1,2])  # default axis=0  return an ordered tensor with selected indices in
            # axis,shape(2,28,28,3),a[1] and a[2]

        tf.gather_nd(a,[0,1,2])      # gather the item with indices of several dimension, shape (3), get the item with
                                     # index 0 in 1st dimension, 1 in second dimension, 2 in third dimension
            tf.gather_nd(a,[[0,1,2]])   # shape (1,3)
            tf.gather_nd(a,[[0,1,2],[0,1,3]])   # get 2 items as tensors with indices, [a[0,1,2], a[0,1,3]]
        tf.boolean_mask(a, mask=[True, True, False, False])   # use boolean mask to pick items needed, here is first 2
                                     # items in first dimension, shape (2,28,28,3), mask array length need same as axis
                                     # length, default axis = 0
            tf.boolean_mask(tf.ones([2,3,4]), mask=[[True, True, False],[False,False,True]])  # 2D mask same shape as
                                     # first 2D of target tensor (2,3)
            tf.boolean_mask(a, a>0) # return a tensor with items value>0

    dimension change
        a = tf.ones([4,28,28,3])
        b = tf.reshape(a, [4,784,3])
        c = tf.reshape(a, [4,-1,3])   # shape (4,784,3), only allow one -1, automatically calculate size

        a = tf.ones([4,20,28,3])
        b = tf.transpose(a)   # shape (3,28,20,4)
        c = tf.transpose(a,[0,3,1,2])   # shape (4,3,20,28)

        b = tf.expend_dims(a, axis=3)   # shape (4,20,28,1,3)
        c = tf.squeeze(tf.zeros([1,2,1,1,3]))     # shape (2,3)
        c = tf.squeeze(tf.zeros([1,2,1,3]), axis=0)     # shape (2,1,3), only able to squeeze dimension size = 1

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
        b2 = tf.tile(tf.ones((2,3,4)),[3,5,1])   # create tensor with shape (6,15,4), repeat x times in each dimension.
                                # same functionality as broadcast, but use more memory


    concat, stack, split
        a = tf.ones([4,20,28,3])
        b = tf.ones([2,20,28,3])
        c = tf.concat([a,b] axis=0)   # concat a and b on axis 0, result shape [6,20,28,3], need have same shape besides
                                      # the contacted axis
        b2 = tf.ones([4,20,28,3])
        c = tf.stack([a,b2] axis=0)    # create a new axis 0 to wrap around 2 tensors, a and b2 must have same shape
                                         # shape (2, 4, 20, 28, 3)
        res = tf.unstack(a, axis=3)     # return a list with 3 tensors with shape (4,28,28)
        res = tf.split(a, axis=0, num_or_size_splits=2)   # return list of 2 same shape (2,28,28,3) tensors
        res = tf.split(a, axis=0, num_or_size_splits=[1,3])   # list of tensors with shape (1,28,28,3),(3,28,28,3)

    math operations
        +  -  *  /     **  pow  square      sqrt     //  %     tf.exp  tf.math.log (e base)     element-wise
        @  matmul (row i in matrix a times column j in matrix b, get result index [i,j] )     matrix-wise
            matrix a shape (4,2,3), matmul matrix b shape (4,3,5)   # return [4,2,5], 4 is batch size
        tf.reduce_mean  tf.max  tf.min  tf.sum   axis-wise

        a.assign_sub(1)  # in place subtract, same as a = a - 1

    Statistic
        Euclidean norm: sqrt(sum((x_i)^2))   x_i: ith component of x, square root of sum of all components square
        Max Norm: max(x_i)   # max of component in all components of x
        L1 Norm: sum(x_i)  # sum of all components of x

        tf.norm
            a = tf.ones([3,3,3])
            tf.norm(a)    # default Euclidean norm(ord=2)    sqrt(1^2 + 1^2 + 1^2 + 1^2)
                # same as tf.sqrt(tf.reduce_sum(tf.square(a)))   return tensor shape of rest axis (3,3) values 1.73
            tf.norm(b, ord=1, axis=1)  #  ord=1 is L1 Norm, axis =1 use all second axis item to form rows
        tf.reduce_min, reduce_max, reduce_mean
            tf.reduce_min(a)  # shape()   return minimum of all item
            tf.reduce_min(a, axis=1)  return min of all second index axis, shape of rest axis (3,3)
        tf.argmax
            tf.argmax(a)   # return list of index with max value, default axis=0 (max index of all axis 0 items,
                           # forming columns using second index)
        tf.equal
            tf.equal(a,b)  # compare each element in same shape tensor a and b, return same shape boolean tensor
            tf.reduce_sum(tf.cast(tf.equal(a,b), dtype=tf.int32))   # count of total correct (same a, b value)
        tf.unique
            res = tf.unique(a)   # return unique items in 1D tensor
            res.y    # return tensor of all unique value
            res.idx     # return a tensor with index of unique value for each value in a

    sort
        b = tf.sort(a, direction='DESCENDING', axis=0)   # default axis=-1, direction='ASCENDING', return tensor fully
                                                        # sorted on specified axis
        idx = tf.argsort(a)   # default axis=-1, 'ASCENDING', return tensor of indices sorted tensor on axis
            # tf.gather(a, idx)
        res = tf.math.top_k(a, 2)  # default k=1
            idx = res.indices  # return tensor of indices of those top k elements
            b = res.values     # return tensor of top k on last axis

    pad
        a = tf.zeros([3,3])
        b = tf.pad(a,[[1,0],[0,1]], constant_values=1)  # return tensor with padding: pad 1 row at front of axis 0, 0
                                                        # row after axis 0, 0 and 1 columns front/after axis 1
                                                        # axis last, default constant_values=0, default mode='CONSTANT'

    tile
        a = tf.zeros([3,3])
        b = tf.tile(a,[2,1])   # copy axis 0 two times, axis 1 one time, return tensor shape (6,3)
        # recommend use tf.broadcast_to   only generate copy during run time, save memory. some function support
            implicit broadcast, ex. a + b will broadcast automatically

    clipping (limit max/min value)
        tf.maximum(a,2)    # return tensor with items less than 2 change to 2
            # tf.nn.relu(a)  same as tf.maximum(a,0)
        tf.minimum(a,8)    # return tensor with items greater than 8 change to 8
        tf.clip(a, 2, 8)   # return tensor with items less than 2 change to 2, greater than 8 change to 8
            # when calculating gradient, clip to a value will change the gradient
            # when calculating gradient, clip to a value will change the gradient

        tf.clip_by_norm(a,15)   # scale the maximum of L2 norm (sqrt(sum(element^2))) to 15, if it's greater than 15,
                                # won't change its gradient
        new_grads, total_norm = tf.clip_by_global_norm(grads,25)  # scale all the tensors with maximum total sum 25,
                                # won't change all input tensors direction, gradient norm is better if within [0-20]
                                # show nan if gradient exploding

    where
        mask = a>0     # return a tensor with same shape as a, if element >0, element become True, otherwise False
        ind = tf.where(mask)    # return tensor with list of indices of elements > 0
        res = tf.gather_nd(a, ind)   # return a tensor with items > 0
            # same as tf.boolean_mask(a, mask)
        res = tf.where(mask, a, b)   # return a tensor, if mask value is true, get same index value from tensor a,
                                     # otherwise from b, mask, a, b must have same shape

    scatter_nd
        res= tf.scatter_nd([[4],[3],[1],[7]], [9,10,11,12], [8])  # first param indices, second value, third shape.
                    # initialize tensor with shape (8) to all zeros, update index 4 value to 9, index 3 value 10,...
        res= tf.scatter_nd([[0],[2]], [[1,1],[1,1]], [3,2])  #update 1st axis index 0, 2 to value 1s [[1,1],[0,0],[1,1]]

    meshgrid
        x, y = tf.linspace(-2, 2, 5), tf.linspace(-2, 2, 5)
        X, Y = tf.meshgrid(x, y)   # both shape [5,5]   X: [[-2. -1.  0.  1.  2.], [-2. -1.  0.  1.  2.]...]
                                   # Y: [[-2. -2. -2. -2. -2.], [-1. -1. -1. -1. -1.]...],
        points = tf.stack([X, Y],axis=2)   # combine X, Y same index element, get a grid of coordinates shape (5,5,2)
        z = tf.math.sin(points[...,0]) + tf.math.sin(points[...,1])
        plt.contour(X, Y, z)   # draw contour map
        plt.imshow(z, origin='lower', interpolation='none')   # draw heat value map for z
        plt.show()

    y = tf.one_hot(y,depth=10, dtype=tf.int32)
    db = tf.data.Dataset.from_tensor_slices((x,y))


    tf.keras (implementation) is not keras(wrapper)
        components inside tf.keras: datasets, layers, losses, metrics, optimizers

    data loading
        keras.datasets: boston housing, mnist, fashion mnist, cifar10/100, imdb. from google host
        from tensorflow.keras import datasets
        def preprocess(x, y)
            x = tf.cast(x, dtype=tf.float32)/255.
            y= tf.one_hot(y,depth=10, dtype=tf.int32)  # y= tf.one_hot(y,depth=10, dtype=tf.int32)
                # no need one_hot if binary classification
            return x, y
        (x, y), (x_test, y_test) = datasets.mnist.load_data()
            # (x, y), (x_test, y_test) = datasets.cifar10.load_data()
        db = tf.data.Dataset.from_tensor_slices((x,y)) # default 1 image a step
            # images, support batch and multi-thread accelerate compare to, for x in dataset grab 1 image one time
        db.map(preprocess)
        db = db.shuffle(10000)   # shuffle data keep x,y as pair. buffer_size, select 10000 items put in buffer
        db = db.batch(128)  # batch 128
        db_iter = iter(db)
        while True:   # non-stop training loop
            try:
                sample = next(db_iter)
            except OutOfRangeError as err:
                db_iter = iter(db)   # when loop through all data, reassign iterator
        or use:
            db2 = db.repeat(2)   # specify repeat train all data 2 times,
               db_inf = db.repeat()  # repeat non-stop
            for (x, y) in db2:  # here will loop through all data 2 times


    Deep Learning break through
        big data, Relu, dropout, batchNorm, ResNet, Xavier Initialization, Caffe/Tensorflow/PyTorch

    Layers
        for single layer
            x = tf.random.normal([4,784])
            net = tf.keras.layers.Dense(512)
            out = net(x)  # this will call build method if no kernel and bias exist
                net.build(input_shape=(None,784)  # None valid for any number
            out.shape   # (4,512)
            net.kernel   # weight (784,512)
            net.bias     # bias (512,)

        for multiple layers
            x = tf.random.normal([4,784])
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation='relu')  # ,input_shape=[None,784]
                tf.keras.layers.Dense(128, activation='relu')
                tf.keras.layers.Dense(10)   # activation='sigmoid' for 2 class classification, 'softmax' for multi-class
            ])
            model.build(input_shape=[None,784])
            model.summary()
            for p in model.trainable_variables:  # weight and bias
                print(p.name, p.shape)

    output activation
        sigmoid (output each element between (0, 1), used for binary classification p(x_0)>0.5 then class 0 )
            b = tf.sigmoid(a)   # return tensor with applying sigmoid to a,  sigmoid = 1/(1+e^-x)
            d/dxσ(x) = σ' = σ(1-σ)     σ:sigmoid
            tf.sigmoid(a), tf.math.sigmoid(a), tf.nn.sigmoid(a)   all the same
        softmax (output each element between 0-1, total sum to 1. used for multi-class classification)
            prob = tf.nn.softmax(a)  # a is output without activation, called score(logits), apply softmax to get
                    # probabilities,   sigma(z) = e^z_j / sum_k(e^z_k)
                    # will make high value class relative percentage stronger, make weak class weaker

        tanh (used mostly in LSTM, RNN) (-1,1)
            tf.tanh(a)    (e^x - e^-x) / (e^x + e^-x) = 2sigmoid(2x) - 1
            tanh(x)' = 1 - (tanh(x))^2

        relu (rectified linear unit)
            f(x) = 0 (kx for leaky relu)   for x<0    x for x>=0
            f' = 0 for x<0    1 for x>=0
            tf.nn.relu(a)
            tf.nn.leaky_relu(a)
    loss metric
        mse:  loss = sum((y-out)^2)/N   # N is entry count, optional divide extra dimension for each element
            # reduce_mean(tf.square(y-out))  =>  sum((y-out)^2)/(N*dim)
            tf.losses.MSE(y, out)
            x, w, b, y = tf.random.normal([2,4]), tf.random.normal([4,3]), tf.zeros([3]), tf.constant([2,0])
            prob = tf.nn.softmax(x@w+b, axis=1)
            loss = tf.losses.MSE(tf.one_hot(y, depth=3),prob))    #  sum((y-out)^2)/N   shape (2,) same as x.shape[0]
        L2 norm: loss = sqrt(sum((y-out)^2))

        entropy:
            H(p) = -sum_i(P(i)*logP(i))   # lower entropy more info, less probability to happen
                                        # ex.[0.5,0.5] probability lowest info, greatest value
                                        # entropy min value is 0 for [0,1,0] one hot encoding
            H(p) = -tf.reduce_sum(a * tf.math.log(a)/tf.math.log(2))
        cross entropy: (min loss is 0)
            H(p,q) = -sum_i(P(i) * logQ(i)) = H(p) + D_kl(P|Q)     D_kl(P|Q)=0 if p and q has same probability
                # if p is one-hot encoding, H(p,q) =-I*log(q_i)     I=0 for p_i=0, I=1 for p_i=1
            loss = tf.losses.categorical_crossentropy([0,1,0,0],[0.01,0.97,0.01,0.01])   # 0.03,  binary_crossentropy
                or metric = tf.losses.CategoricalCrossentropy()
                   loss = metric([0, 1, 0, 0], [0, 0.0001, 0, 0.9999])
            better than mse+sigmoid (gradient vanish, converge slower)
            softmax, then crossentropy might cause unstable state, usually use following:
                logits = h@w+b  # last hidden layer generate output layer without activation function
                loss = tf.losses.categorical_crossentropy([0,1,0,0], logits, from_logits=True)
                                                                        # combine softmax, cross entropy
                                # here logits is result without applying cross entropy
                    # prob = tf.math.softmax(logits, axis=1)
                    # loss = tf.losses.categorical_crossentropy([0,1,0,0], prob)  # unstable result, not recommended

    gradient descent
        derivative:  lim x->0 (∆y)
        partial derivative: for different axis    ∂f / ∂x,  ∂f / ∂y
        gradient: combine of all axis partial derivative  ∇f = (∂f/∂x_1; ∂f/∂x_2;... ∂f/∂x_n), pointing to direction
            with fastest increase f
            θ_t+1 = θ_t - α∇f(θ_t)     α: learning rate

            with tf.GradientTape() as tape:     # must use float for variables
                tape.watch([w1, w2])     # not required if w is a tf.Variable
                y = x1 * w1 + x2 * w2    # put all calculation in tape to track gradients
                grad = tape.gradient(y, [w1, w2])   # grad = [dw1, dw2] return list of tensors for gradients
                                                    # grad = tape.gradient(y, w1)  return tensor for gradient
            # tape.gradient can only call once, then will release resource unless use:
                with tf.GradientTape(persistent=True) as tape:

            # 2nd order gradient
                x, w, b = tf.Variable(1.0), tf.Variable(2.0), tf.Variable(3.0)
                with tf.GradientTape() as t1:
                    with tf.GradientTape() as t2:
                        y = x * w + b
                    dy_dw, dy_db = t2.gradient(y, [w, b])
                d2y_dw2 = t1.gradient(dy_dw, w)
                print(d2y_dw2)


    tensorboard
        cd to work directory, tensorboard --logdir resources/logs    # logs is log directory

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join('resources', 'logs',current_time),
        summary_writer = tf.summary.create_file_writer(log_dir)

        img = tf.reshape(img, [1,28,28,1])
        imgs = tf.reshape(imgs, [-1,28,28,1])
        imgs_one = image_grid(imgs)
        with summary_writer.as_default():
            tf.summary.scalar('loss', float(loss), step=epoch)   # show scalar as y axis against epoch x axis
            tf.summary.image('sample image', img, step=0)  # show 1 image
            tf.summary.image('sample images', imgs, max_output=25, step=step)   # show batch images with 1 img a spot
            tf.summary.image('in one plot', plt_to_image(imgs_one), step=step)

            # def image_grid(images):
                  figure = plt.figure(figsize=(10,10))
                  for i in range(25):
                     plt.subplots(5,5,i+1, title='name')
                     plt.imshow(images[i], cmap=plt.cm.binary)
                     return figure

            # def plot_to_image(figure):
                  buf = io.BytesIO()
                  plt.savefig(buf, format='png')  # save image in memory  .Mean()
                  plt.close(figure)
                  buf.seek(0)
                  image = tf.image.decode_png(buf.getvalue(), channels=4)
                  image = tf.expand_dims(image,0)
                  return image

    Metrics
        from tf.keras import metrics
        acc_m = metrics.Accuracy()            # initialize metrics
            # loss_m = metrics.Mean()
        acc_m.update_state(y, pred)           # update_state
            # loss_m.update_state(loss)
        print(step, 'acc:' acc_m.result().numpy())     # print result
        acc_m.reset_states()                   # clear previous stored acc value

    keras train fit predict
        model.compile(optimizer=optimizers.Adam(learning_rate=0.01),
                  loss=tf.losses.CategoricalCrossEntropy(from_logits=True), metrics=['accuracy'])
        model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_steps=2)
            # validate during training every 2 epochs validate once
        model.evaluate(test_dataset)  # validate after training

        test_sample = next(iter(test_dataset))
        test_x, test_y = test_sample[0], test_sample[1]
        pred = model.predict(test_x)  # shape (b,10)
        y = tf.argmax(test_y, axis=1)
        pred = tf.argmax(pred, axis=1)
        print(y, pred)


        x_train, x_val = tf.split(x, num_or_size_splits=[50000,10000])
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        loss_m = metrics.Mean()
        model = tf.keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10)
        ])
        optimizer = optimizers.Adam(learning_rate=1e-3)
        for epoch in range(10):
            for step, (x, y) in enumerate(train_dataset):
                x = tf.reshape(x, [-1, 28 * 28])
                with tf.GradientTape() as tape:
                    logits = model(x)
                    y_onehot = tf.one_hot(y, depth=10)
                    loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                    loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))

                    loss_m.update_state(loss_ce)

                grads = tape.gradient(loss_ce, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if step % 100 == 0:
                    print(epoch, step, 'loss: ', float(loss_ce), loss_m.result().numpy())
                    loss_m.reset_states()

            total_correct, total_num = 0, 0
            acc_m.reset_states()
            for x, y in test_dataset:
                x = tf.reshape(x, [-1, 28 * 28])
                logits = model(x)
                prob = tf.nn.softmax(logits, axis=1)
                pred = tf.cast(tf.argmax(prob, axis=1),dtype=tf.int32)
                correct = tf.equal(y, pred)
                correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
                total_correct += int(correct)
                total_num += x.shape[0]

            acc = total_correct / total_num
            acc_m.update_state(y, pred)
            print('test acc', acc,  acc_m.result().numpy())

    self defined layer
        in order to use keras.Sequential to ease the getting gradient update for trainable_variable, we need extend
            usage of keras.layers.Layer and keras.Model

        model.build(input_shape=[None,28*28])  model.summary()
            # or model(train_dataset)   # inner run model.__call__(train_dataset)
        Sequential is the child class of keras.Model, which have compile/fit/evaluate/predict functions
        self defined layer should extend keras.layers.Layer

            class MyDense(layers.Layer):  # self define layer extend from tensorflow.keras.layers.Layer
                def __init__(self, inp_dim, outp_dim):  # need override __init__ and call method
                    super(MyDense, self).__init__()
                    self.kernel = self.add_weight('w', [inp_dim, outp_dim])  # don't use tf.constant()
                    self.bias = self.add_weight('b', [outp_dim])
                        # use add_variable so all layers variable can be organized and optimized together
                def call(self, inputs, training=None):  # default training and test apply same logic
                    out = inputs @ self.kernel + self.bias
                    return out

            class MyModel(keras.Model):
                def __init__(self):  # need override __init__ and call method
                    super(MyModel, self).__init__()
                    self.fc1 = MyDense(28*28,256)
                    self.fc2 = MyDense(256,64)
                    self.fc3 = MyDense(64,10)
                def call(self, inputs, training=None):  # default training and test apply same logic
                            # MyModel(x) or MyModel(x, training=True)  train mode. MyModel(x, training=False)  test mode
                    x = self.fc1(inputs)
                    x = tf.nn.relu(x)
                    x = self.fc2(x)
                    x = tf.nn.relu(x)
                        # x = x - 1    # able to do flexible logic that Sequential can't do
                    x = self.fc3(x)
                    return x
            model = MyModel()
            model.compile/fit/evaluate/predict  same as above

    save, load model
        1. save / load  weights
        2. save / load entire model
        3. saved_model to uniform format, can be read by other program

        1.  save weights
            model.save_weights('../resources/checkpoint/weights.ckpt')
            del model   # simulate model gone in memory
            model = Sequential() model.compile()   # same as above, need build network and compile
                model = create_model()
            model.load_weights('../resources/checkpoint/weights.ckpt')
            loss, acc = model.evaluate(test_images, test_labels)

        2. save entire model
            model.save('../resources/checkpoint/model.h5')
            del model
            model = tf.keras.models.load_model('../resources/checkpoint/model.h5')  # no need create network
            loss, acc = model.evaluate(test_images, test_labels)

        3. save model under protocol can be utilized by other program (production environment)
            tf.save_model.save(model,'../resources/saved_model/')
            del model
            imported = tf.saved_model.load('../resources/saved_model/')
            model = imported.signatures["serving_default"]
            loss, acc = model.evaluate(test_images, test_labels)

    add validation set
        # split train set to train and validation set
            x_train, x_val = tf.split(x, num_or_size_splits=[50000,10000])
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            model.fit(train_dataset, epochs=10, validation_data=val_dataset, validation_step=2)
            model.evaluate(test_dataset)

        k-fold validation
            for epoch in range(10):
                idx = tf.range(60000)
                idx = tf.random.shuffle(idx)
                x_train, y_train = tf.gather(x,idx[:50000]), tf.gather(y,idx[:50000])
                x_val, y_val = tf.gather(x,idx[-10000:]), tf.gather(y,idx[-10000:])

            or:
                model.fit(train_dataset, epochs=10,  validation_split=0.1, validation_steps=2)

    reduce overfitting:
        get more data, add dropout, reduce model complexity(choose simpler network, add regularization), early stop

    regularization:
        L2 norm (loss function add + 0.5λ*||W||^2)) or L1 norm (loss function add + λ*sum_i(θ_i))
        during define model layers add:
            tf.keras.layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
                # here using l2 loss, with lambda=0.001

        or use gradient tape to make more flexible:
            for step, (x,y) in enumerate(train_dataset):
                with tf.GradientTape()  as tape:
                    loss = tf.reduce_mean(tf.losses.categotical_crossentropy(y_onehot, out, from_logits=True))
                    loss_regularization = []
                    for p in model.trainable_variables:  #[:4] first 4 weights
                        loss_regularization.append(tf.nn.l2_loss(p))
                    loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
                    loss = loss + 0.0001 * loss_regularization
                grads = tape.gradient(loss, model.trainable_variables)

    momentum
        ω_(k+1) = ω_k - α∇f(ω_k)   # sgd
        ω_(k+1) = ω_k - αz_(k+1)   z_(k+1) = β*z_k + ∇f(ω_k)

        optimizer = SGD(learning_rate=0.02, momentum=0.9)
        optimizer = RMSprop(learning_rate=0.02, momentum=0.9)
        optimizer = Adam(learning_rate=0.02, beta_1=0.9, beta_2=0.999)

    learning rate decay
        for epoch in range(100):
            optimizer.learning_rate = 0.2 * (100 - epoch) / 100   # update optimizer.learning_rate

    early stop
        stop when test accuracy start rising

    dropout
        don't randomly ignore(don't use) some nodes in hidden  layers during training. reduce overfitting

        tf.keras.layers.Dropout(0.4) # 40% dropout, more dropout, harder training, less over fitting, better
                                        # generalization
        out = model(x, training=True)  # True during train and false(default) for validation and test



    cnn
    feature map: data from each layer, include input layer (width * height * channel(filter count))
        layer = layers.Conv2D(4, kernel_size=5, strides=1, padding='valid')
            # 4 filters, no need specify depth(channel) of each kernel
            # 'valid': no padding,   'same': padding till result same shape
        out = layer(x)   # inside use function __call__ ->  call() method
        layer.kernel    # return tensor with (shape, dtype, numpy)
        layer.bias      # return tensor with (shape, dtype, numpy)

        or use flexible, deep layer implementation, not recommended, hard to maintain
            print(x.shape)    # [1,32,32,3]
            w = tf.random.normal([5,5,3,4])   # height, width, channel, kernel count
            b = tf.zeros([4])
            out = tf.nn.conv2d(x, w, strides=1, padding='VALID')     # tensor [1,28,28,4]

    pooling
        x.shape   # [1,14,14,4]
        layer = layers.MaxPool2D(2, strides=2)  # size
        out = layer(x)   # [1,7,7,4]
        # or use function
        out = tf.nn.max_pool2d(x, 2, strides=2, padding='VALID')

        up sampling
            layer = layers.UpSampling2D(size=3)  # change input shape size to 3 times as output
            out = layer(x)  # [1,42,42,4]

        relu
            tf.nn.relu(x)
            or  layers.ReLU()(x)

    Resnet block
        class BasicBlock(layers.Layer):
            def __init__(self,filter_num, stride=1):
                super(BasicBlock, self).__init()

                self.conv1 = layers.Conv2D(filter_num, (3,3), strides=stride, padding='same')
                self.bn1 = layers.BatchNormalization()
                self.relu = layers.Activation('relu')
                self.conv2 = layers.Conv2D(filter_num, (3,3), strides=1, padding='same')
                self.bn2 = layers.BatchNormalization()
                if stride != 1:
                    self.downsample = Sequential()
                    self.downsample.add(layers.Conv2D(filter_num, (1, 1), stride=stride))
                    self.downsample.add(layers.BatchNormalization())
                else:
                     self.downsample = lambda x: x

                self.stride = stride

            def call(self, inputs, training=None):
                residual = self.downsample(inputs)
                conv1 = self.conv1(inputs)
                bn1 = self.bn1(conv1)
                relu1 = self.relu(bn1)
                conv2 =  self.conv2(relu1)
                bn2 =  self.bn2(conv2)

                add = layers.add([bn2,residual])
                out = self.relu(add)
                return out

    RNN
        h_t = tanh(x_t * w_xh + h_t-1 * w_hh + b)
            # [batch, feature len] * [feature len, hidden len] + [batch, hidden len] * [hidden len, hidden len]
            # x: (batch, seq len, feature len)
        y_t = w_o * h_t

        apply gradient clipping (set threshold) to avoid gradient explode
        use LSTM for gradient vanish

        x = tf.random.normal([4,80,100])    # 4 batch, 80 sentence len, 100 features per word
        xt0 = x[:,0,:]
        cell = layers.SimpleRNNCell(64)   # 64: hidden len
        cell2 = layers.SimpleRNNCell(64)  # second depth layer of RNN
            # cell.build(input_shape=(None, 100))   # 100: feature len
            # cell.trainable_variables  : w_xh, w_hh,   b shape(64,)
        state0, state1 = [tf.zeros(batch, units)],[tf.zeros(batch, units)]    # (4,64)
        for word in tf.unstack(x, axis=1):  # word shape (b,100)
            out0, state0 = cell(word, state0)
                # out0 shape: (4,64)  same as state0    state0 is a list of tensor
            out1, state1 = cell(out0, state1)

        grads = [tf.clip_by_norm(g,15) for g in grads]   # clip gradient to max 15


        or use
            rnn = keras.Sequential([
                layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
                    # units 64, return sequence if not output layer. unroll speed up training, but consume more memory,
                    # only for short sequence, default use_bias=True
                layer.SimpleRNN(units, dropout=0.5, unroll=True),
            ])
            x = self.rnn(x)  # last sequence time and last layer output


    # distribute training
        tf.distribute.MirroredStrategy supports synchronous distributed training on multiple GPUs on one machine. It
            creates one replica per GPU device. Each variable in the model is mirrored across all the replicas.
        tf.distribute.MultiWorkerMirroredStrategy is very similar to MirroredStrategy. It implements synchronous
            distributed training across multiple workers, each with potentially multiple GPUs.
        Parameter server training is a common data-parallel method to scale up model training on multiple machines.
            A parameter server training cluster consists of workers and parameter servers. Variables are created on
            parameter servers and they are read and updated by workers in each step.

        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

        communication_options = tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.
            experimental.CommunicationImplementation.NCCL)
        strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)

        strategy = tf.distribute.experimental.ParameterServerStrategy(tf.distribute.cluster_resolver.
            TFConfigClusterResolver(), variable_partitioner=variable_partitioner)
        coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)
        with strategy.scope():
            https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/ParameterServerStrategy

        yahoo TensorflowOnSpark framework combine spark and tensorflow
        distributed ingest: SparkSQL + spark-mongo-connector
        distributed train: TensorflowOnSpark + Tensorflow
        distributed evaluation: spark Mlib

        spark = SparkSession.builder.appName('Distribute ML').getOrCreate()
        sc = spark.sparkContext
        df.createOrReplaceTempView("IRIS")  #  create a temporary table
        df = spark.sql("SELECT img_content, label from images where app_id=1")
        rdd = df.rdd.map(lambda x: (bytes(x[0]), numpy.asarray(x[1], numpy.uint16)))
        rdd.persist(StorageLevel.MEMORY_AND_DISK_2)


'''
import datetime
import glob
import os
import time

import tensorflow as tf
from PIL import Image
from sklearn.externals._pilutil import toimage
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, metrics
from scipy import misc

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import timeit

from dataset import disk_image_batch_dataset, make_anime_dataset


def top_k_accuracy(output, target, topk=(1,)):
    maxk = max(topk)  # consider top k
    batch_size = output.shape[0]
    pred = tf.top_k(output, maxk).indices
    pred = tf.transpose(pred, perm=[1, 0])
    target_ = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target_)
    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)  # consider top k rows if any equal to
        # broadcasted target
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k / batch_size)
        res.append(acc)
        return res
    # top_k_accuracy(output, target, topk=(1,2,3,4,5,6))   output shape(10,6)  target shape [10], 10 entries 6 category


def mnist_digits():
    (x, y), (x_val, y_val) = datasets.mnist.load_data()
    x = tf.convert_to_tensor(x, dtype=tf.float32) / 255
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)  # one hot encoding with 10 columns, with label class column 1, 9 columns 0
    print(x.shape, y.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.batch(128)  # default 1 image a step, here batch 128 images

    model = keras.Sequential([
        layers.Dense(512, activation='relu'),
        # layer = layers.Dense(512, activation='relu') layer.kernel.shape, layer.bias.shape
        layers.Dense(256, activation='relu'),
        layers.Dense(10)
    ])
    optimizer = optimizers.SGD(learning_rate=0.001)

    def train_epoch(epoch):
        for step, (x, y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                x = tf.reshape(x, (-1, 28 * 28))  # [b,28,28] => [b, 784]
                out = model(x)  # output  [1,10]
                loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]  # loss function

            grads = tape.gradient(loss, model.trainable_variables)  # w, b in each layers
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss: ', loss.numpy())

    def train():
        for epoch in range(30):
            train_epoch(epoch)

    train()


class my_call_back(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.2:
            print("\nLoss is good enough, cancelling training")
            self.model.stop_training = True


def example1():
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])  # sequential 1 layer 1 neuron
    model.compile(optimizer='sgd', loss='mean_squared_error')

    x = np.array([-1.0, 0, 1, 2, 3, 4], dtype=float)
    y = np.array([-3.0, -1, 1, 3, 5, 7], dtype=float)
    model.fit(x, y, epochs=500)
    print(model.predict([10]))


def mnist_fashion():
    callbacks = my_call_back()
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()  # 28*28

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
        keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # filters count, filter shape
        keras.layers.MaxPool2D(2, 2),  # max pooling
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),  # 128 nodes in the hidden layer
        keras.layers.Dense(10, activation='softmax')  # 10 class
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()
    model.fit(X_train, y_train, epochs=5)  # , callbacks=[callbacks]
    y_pred = model.predict(X_test)  # return probability array for each class

    print(model.evaluate(X_train, y_train))

    m = tf.keras.metrics.Accuracy()
    m.update_state(y_test, np.argmax(y_pred, axis=1))

    print(m.result().numpy())  # accuracy 0.887   0.913

    f, axes = plt.subplots(3, 4)
    FIRST_IMAGE = 4
    SECOND_IMAGE = 7
    THIRD_IMAGE = 26
    CONVOLUTION_NUMBER = 4
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    for x in activation_model.predict(X_test[FIRST_IMAGE].reshape(1, 28, 28, 1)):
        print(x.shape)

    for x in range(0, 4):
        f1 = activation_model.predict(X_test[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]

        axes[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axes[0, x].grid(False)
        f2 = activation_model.predict(X_test[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
        axes[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axes[1, x].grid(False)
        f3 = activation_model.predict(X_test[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
        axes[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axes[2, x].grid(False)
    plt.show()


def convolution_pooling():
    img = misc.ascent()
    plt.gray()  # change the cmap to gray, do not convert color image to gray scale
    # plt.axis('off')
    plt.grid(False)
    plt.imshow(img)
    # plt.show()

    img_tran = np.copy(np.array(img))
    height, width = img_tran.shape[0], img_tran.shape[1]
    weight = 1
    filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # activates horizontal lines
    # filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # activates straight lines
    # apply filter to do convolution, sum of (multiply same location value of filter and image) + bias
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            box = img[i - 1:i + 2, j - 1:j + 2]
            box[1, 0] = img[i, j - 1]
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
    height_p, width_p = img_tran.shape[0] // 2, img_tran.shape[1] // 2
    img_pool = np.zeros((height_p, width_p))
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            pix = np.max(img_tran[i:i + 2, j:j + 2])
            img_pool[i // 2, j // 2] = pix
    plt.gray()
    plt.imshow(img_pool)
    plt.show()


def cifar102():
    callback = my_call_back()
    X_train, X_test, y_train, y_test = [], [], [], []
    count = 0
    for category in os.listdir('../resources/data/cifar10/train'):
        for name in os.listdir(os.path.join('../resources/data/cifar10/train', category)):
            temp = plt.imread(os.path.join('../resources/data/cifar10/train/', category, name))
            # temp = (0.299 * temp[:, :, 0] + 0.587 * temp[:, :, 1] + 0.114 * temp[:, :, 2]) / 3
            X_train.append(np.array(temp))
            y_train.append(count)
        count += 1
    count = 0
    for category in os.listdir('../resources/data/cifar10/test'):
        for name in os.listdir(os.path.join('../resources/data/cifar10/test', category)):
            temp = plt.imread(os.path.join('../resources/data/cifar10/test/', category, name))
            # temp = (0.299 * temp[:, :, 0] + 0.587 * temp[:, :, 1] + 0.114 * temp[:, :, 2]) / 3
            X_test.append(temp)
            y_test.append(count)
        count += 1
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)









os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only show error log from tensorflow

def test():
    x, w, b, y = tf.random.normal([2, 4]), tf.random.normal([4, 3]), tf.zeros([3]), tf.constant([2, 0])
    prob = tf.nn.softmax(x @ w + b, axis=1)
    loss = tf.losses.MSE(tf.one_hot(y, depth=3), prob)
    print(loss.shape)

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


def save_images(imgs, name):
    new_im = Image.new('L',(280,280))
    index = 0
    for i in range(0,280,28):  # put 100 28*28 img into one 280*280 img
        for j in range(0,280,28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(name)


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.   # for better result change input range to [-1,1]
    # x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def mnist_from_scratch():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    print('a',x.shape, y.shape, type(x), type(y))  # (60000,28,28)  (60000,)
    #x = tf.convert_to_tensor(x, dtype=tf.float32)/255
    #y = tf.convert_to_tensor(y, dtype=tf.int32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

    print(tf.reduce_min(x)) # tf.reduce_max(x,axis=0)
    train_dataset = tf.data.Dataset.from_tensor_slices((x,y)) # default 1 image a step
    train_dataset = train_dataset.map(preprocess).shuffle(10000).batch(128) # here batch 128 images
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

    train_iter = iter(train_dataset)
    sample = next(train_iter)
    print('batch: ', sample[0].shape, sample[1].shape)  # # (128,28,28)  (128,)

    # [b, 784] => [b, 256] => [b, 128] => [b, 10]
    w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))  # need to be variable in order to get auto gradient
    b1 = tf.Variable(tf.zeros([256]))
    w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1)) # cause gradient explode if std too large
    b2 = tf.Variable(tf.zeros([128]))
    w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))
    lr = 1e-3
    '''
    for epoch in range(30):
        for step, (x, y) in enumerate(train_dataset):
            x = tf.reshape(x, [-1, 28*28])   # shape (b, 784)

            with tf.GradientTape() as tape:
                h1 = x @ w1 + tf.broadcast_to(b1, [x.shape[0],256])       # auto broadcast or explicit   # shape (b,256)
                        # hidden layer 256 nodes, w1 784 elements for each node, total 784*256+256(bias) param for layer
                h1 = tf.nn.relu(h1)   # relu non-linear activation
                h2 = h1 @ w2 + b2    # (b, 256)@(256, 125)  shape (b, 128)
                h2 = tf.nn.relu(h2)
                out = h2 @ w3 + b3    # shape (b, 10)

                y_onehot = tf.one_hot(y, depth=10)   # one hot encoding with 10 columns
                # mse = mean(sum(y-out)^2)
                loss = tf.square(y_onehot - out)   # shape [b,10]
                loss = tf.reduce_mean(loss)  # shape ()

            train_variables = [w1, b1, w2, b2, w3, b3]
            grads = tape.gradient(loss, train_variables)

            for i, var in enumerate(train_variables):
                var.assign_sub(lr * grads[i])   # in place replacement or assign var = tf.Variable(var - lr * grads[i])

            if step % 100 == 0:
                print(epoch, step, 'loss: ', float(loss))

        total_correct, total_num = 0, 0
        for step, (x, y) in enumerate(test_dataset):
            x = tf.reshape(x, [-1, 28 * 28])  # shape (b, 784)
            h1 = tf.nn.relu(x @ w1 + b1)
            h2 = tf.nn.relu(h1 @ w2 + b2)
            out = h2 @ w3 + b3  # shape (b, 10)

            prob = tf.nn.softmax(out, axis=1)  # shape (b, 10)
            pred = tf.argmax(prob, axis=1)  # shape (b,)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print('test acc', acc)

    
    '''


    model = tf.keras.Sequential([
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(10)
    ])
    model.build(input_shape=[None,28*28])
    model.summary()
        # or model(train_dataset)   # inner run model.__call__(train_dataset)

    optimizer = optimizers.Adam(learning_rate=1e-3)
    ''' 
    acc_m = metrics.Accuracy()
    loss_m = metrics.Mean()
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('..','resources', 'logs', current_time),
    summary_writer = tf.summary.create_file_writer(log_dir)


    sample_img = tf.reshape(sample[0][0],[1,28,28,1])
    print(sample_img.shape)
    with summary_writer.as_default():
        tf.summary.image('sample image', sample_img, step=0)  # show 1 image

    for epoch in range(10):
        for step, (x, y) in enumerate(train_dataset):
            x = tf.reshape(x, [-1, 28 * 28])
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
                
                loss_m.update_state(loss_ce)

            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss: ', float(loss_ce), loss_m.result().numpy())
                with summary_writer.as_default():
                    tf.summary.scalar('train-loss', float(loss_ce), step=epoch)# show scalar as y axis against epoch x axis
                loss_m.reset_states()

        total_correct, total_num = 0, 0
        acc_m.reset_states()
        for x, y in test_dataset:
            x = tf.reshape(x, [-1, 28 * 28])
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.cast(tf.argmax(prob, axis=1),dtype=tf.int32)
            correct = tf.equal(y, pred)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            total_correct += int(correct)
            total_num += x.shape[0]
            
            val_imgs = tf.reshape(x[:25], [-1,28,28,1])
            with summary_writer.as_default():
                tf.summary.image('val-images', val_imgs, max_outputs=25, step=step)
                
        acc = total_correct / total_num
        acc_m.update_state(y, pred)
        print('test acc', acc,  acc_m.result().numpy())
        with summary_writer.as_default():
            tf.summary.scalar('test accuracy', acc, step=epoch)  # show scalar as y axis against epoch x axis
    '''
    print('a', x.shape, y.shape, type(x), type(y))  # (60000,28,28)  (60000,)
    train_dataset = tf.data.Dataset.from_tensor_slices((x, tf.one_hot(y,depth=10)))  # default 1 image a step
    train_dataset = train_dataset.map(preprocess).shuffle(10000).batch(128)  # here batch 128 images
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, tf.one_hot(y_test,depth=10)))  # default 1 image a step
    test_dataset = train_dataset.map(preprocess).shuffle(10000).batch(128)  # here batch 128 images


def preprocess2(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1.  # for better result change input range to [-1,1]
    y = tf.cast(y, dtype=tf.int32)
    return x, y


class MyDense(layers.Layer):  # self define layer extend from tensorflow.keras.layers.Layer
    def __init__(self, inp_dim, outp_dim):  # need override __init__ and call method
        super(MyDense, self).__init__()
        self.kernel = self.add_weight('w', [inp_dim, outp_dim])  # don't use tf.constant()
        # self.bias = self.add_weight('b', [outp_dim])

    def call(self, inputs, training=None):  # default training and test apply same logic
        out = inputs @ self.kernel
        return out


class MyModel(keras.Model):
    def __init__(self):  # need override __init__ and call method
        super(MyModel, self).__init__()
        self.fc1 = MyDense(32*32*3, 256)
        self.fc2 = MyDense(512, 256)
        self.fc3 = MyDense(256, 128)
        self.fc4 = MyDense(128, 64)
        self.fc5 = MyDense(64, 10)

    def call(self, inputs, training=None):  # default training and test apply same logic
        x = tf.reshape(inputs, [-1, 32*32*3])
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x


def cifar10():
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    #tf.config.gpu.set_per_process_memory_fraction(0.75)
    #tf.config.Gpu.sset_per_process_memory_growth(True)
    tf.random.set_seed(8)
    (x, y), (x_val, y_val) = datasets.cifar10.load_data()   # y shape (50000,1,10)
    y, y_val = tf.squeeze(y), tf.squeeze(y_val)
    y, y_val = tf.one_hot(y, depth=10), tf.one_hot(y_val, depth=10)
    print('datasets', x.shape, y.shape, x_val.shape, y_val.shape)
    BATCH_SIZE = 2

    train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
    train_dataset = train_dataset.map(preprocess2).shuffle(1000).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = test_dataset.map(preprocess2).batch(BATCH_SIZE)

    sample = next(iter(train_dataset))
    print('batch:', sample[0].shape, sample[1].shape)

    '''
    model = MyModel()
    model.compile(optimizer=optimizers.Adam(learning_rate=0.01),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_steps=2)
    model.summary()
        # validate during training every 2 epochs validate once
    model.evaluate(test_dataset)
    model.save_weights('../resources/model/weights.ckpt')
    del model

    model = MyModel()

    model.compile(optimizer=optimizers.Adam(learning_rate=0.01),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.load_weights('../resources/model/weights.ckpt')
    model.evaluate(test_dataset)

    '''

    conv_layers = [
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(10, activation=None)
    ]


    fc_net = tf.keras.Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(10, activation=None),
    ])

    conv_net = tf.keras.Sequential(conv_layers)
    conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 512])
    variables = conv_net.trainable_variables + fc_net.trainable_variables
    optimizer = optimizers.Adam(learning_rate=1e-4)


    for epoch in range(10):
        for step, (x,y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                out = conv_net(x)
                out = tf.reshape(out, [-1, 512])
                logits = fc_net(out)
                #y_onehot = tf.one_hot(y, depth=10)
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y, logits, from_logits=True))
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

        total_num, total_correct = 0, 0
        for x, y in test_dataset:
            out = conv_net(x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_net(out)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            y_ = tf.argmax(y, axis=1)
            y_ = tf.cast(y_, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y_), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print('test acc', acc)



class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
            self.downsample.add(layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

        self.stride = stride

    def call(self, inputs, training=None):
        residual = self.downsample(inputs)
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        output = layers.add([out, residual])
        output = self.relu(output)
        return output

class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=100):  # [2,2,2,2]
        super(ResNet, self).__init__()
        self.preprocess = tf.keras.Sequential([layers.Conv2D(64, (3, 3), strides=(1,1)),
                                               layers.BatchNormalization(),
                                               layers.Activation('relu'),
                                               layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')
                                              ])
        self.layer1 = self.build_resblock(64,layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)  # output shape [b, 512, h, w]

        self.avgpool = layers.GlobalAveragePooling2D()   # output shape [b, 512]
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.preprocess(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)  # output shape [b, 100]
        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = tf.keras.Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))  # only 1st able to downsample

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

def resnet18():
    return ResNet([2,2,2,2])  # 1 + 2 * 2 + 2 * 2 + 2 * 2 + 2 * 2 + 1

def resnet34():
    return ResNet([3,4,6,3])

def cifar100():
    tf.random.set_seed(8)
    (x, y), (x_val, y_val) = datasets.cifar100.load_data()  # y shape (50000,1,10)
    y, y_val = tf.squeeze(y, axis=1), tf.squeeze(y_val, axis=1)
    #y, y_val = tf.one_hot(y, depth=10), tf.one_hot(y_val, depth=10)
    print('datasets', x.shape, y.shape, x_val.shape, y_val.shape)
    BATCH_SIZE = 128

    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.map(preprocess2).shuffle(1000).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = test_dataset.map(preprocess2).batch(BATCH_SIZE)

    sample = next(iter(train_dataset))
    print('batch:', sample[0].shape, sample[1].shape)

    model = resnet18()
    model.build(input_shape=(None, 32,32,3))
    model.summary()
    optimizer = optimizers.Adam(learning_rate=1e-4)

    for epoch in range(50):
        for step, (x,y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=100)
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

        total_num, total_correct = 0, 0
        for x, y in test_dataset:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print('test acc', acc)

VOCABULARY_SIZE = 10000
MAX_SENTENCE_LENGTH = 80
EMBEDDING_SIZE = 100
BATCH_SIZE = 64

class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        self.state0 = [tf.zeros([BATCH_SIZE, units])]  # hidden state 0
        self.state1 = [tf.zeros([BATCH_SIZE, units])]  # hidden state 0
        self.embedding = layers.Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH)
            # transfer text to embedding  (b,80) => (b,80,100)
        self.rnn_cell0 = layers.SimpleRNNCell(units , dropout=0.4)   # units is feature length (nodes count)
            # can't use dropout for SimpleRNNCell, unless tf.compat.v1.disable_eager_execution()
            # hidden layer shape (b, 64)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.4)
        self.outlayer = layers.Dense(1)   # shape (b, 64) => (b, 1)

    def call(self, inputs, training=None):
        x = self.embedding(inputs)   # shape(b, 80) => (b,80,100)
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):   # word (b,100)
            out0, state0 = self.rnn_cell0(word, state0, training)  # don't activate dropout during test act differently
            out1, state1 = self.rnn_cell1(out0, state1, training)
        x = self.outlayer(out1)  # shape (b,64) => (b,1)
        prob = tf.sigmoid(x)
        return prob

class MyLSTM(keras.Model):
    def __init__(self, units):
        super(MyLSTM, self).__init__()
        self.state0 = [tf.zeros([BATCH_SIZE, units]), tf.zeros([BATCH_SIZE, units])]  # c0 and h0
        self.state1 = [tf.zeros([BATCH_SIZE, units]), tf.zeros([BATCH_SIZE, units])]
        self.embedding = layers.Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH)
            # transfer text to embedding  (b,80) => (b,80,100)
        self.rnn_cell0 = layers.LSTMCell(units , dropout=0.4)   # units is feature length (nodes count)
            # can't use dropout for SimpleRNNCell, unless tf.compat.v1.disable_eager_execution()
            # hidden layer shape (b, 64)
        self.rnn_cell1 = layers.LSTMCell(units, dropout=0.4)
        self.outlayer = layers.Dense(1)   # shape (b, 64) => (b, 1)

    def call(self, inputs, training=None):
        x = self.embedding(inputs)   # shape(b, 80) => (b,80,100)
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):   # word (b,100)
            out0, state0 = self.rnn_cell0(word, state0, training)  # don't activate dropout during test act differently
            out1, state1 = self.rnn_cell1(out0, state1, training)
        x = self.outlayer(out1)  # shape (b,64) => (b,1)
        prob = tf.sigmoid(x)
        return prob

def imdb():
    tf.random.set_seed(8)
    assert tf.__version__.startswith('2.')
    tf.compat.v1.disable_eager_execution()   # solve SimpleRNNCell dropout  issue
    UNITS = 64

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=VOCABULARY_SIZE)
        # only encode most common 10000 words, uncommon words treat as same word
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_SENTENCE_LENGTH)
        # padding sentence if length less than 80, or trim long sentence to 80 words,  shape (total count, 80)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_SENTENCE_LENGTH)


    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE, drop_remainder=True) # drop last batch remainder
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
    print(x_train.shape,y_train.shape)
    #sample = next(iter(train_dataset))
    #print('batch:', sample[0].shape, sample[1].shape)
    t0 = time.time()
    '''
    #model = MyRNN(UNITS)
    model = MyLSTM(UNITS)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.fit(train_dataset, epochs=4, validation_data=test_dataset)
    model.evaluate(test_dataset)
    '''
    model = keras.Sequential([
        layers.Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH),
        #layers.SimpleRNN(64, dropout=0.5, return_sequences=True),
        #layers.SimpleRNN(64, dropout=0.5),
        layers.LSTM(64, dropout=0.5, return_sequences=True),
        layers.LSTM(64, dropout=0.5),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss=tf.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model.fit(train_dataset, epochs=4, validation_data=test_dataset)
    model.evaluate(test_dataset)

    t1 = time.time()
    print('time cost: %d', t1-t0)



h_dim = 20    # feature size for z, reduce dimension of image to 20
z_dim = 10   # dimension for mean and std for z  for Variational auto encoder

class AE(keras.Model):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None):
        h = self.encoder(inputs)   # shape (784,10) => (b,10)
        x_gen = self.decoder(h)   # shape  (b,10) => (784,10)
        return x_gen

class VAE(keras.Model):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(z_dim)
        self.fc3 = layers.Dense(z_dim)

        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    def encoder(self, x):
        h = tf.nn.relu(self.fc1(x))
        means = self.fc2(h)
        log_var = self.fc3(h)   # log variance (-inf, inf), auto learned better than variance (0,inf)
        return means, log_var

    def decoder(self, z):
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)
        return out

    def reparameterize(self, means, log_var):
        eps = tf.random.normal(log_var.shape)
        std = tf.exp(log_var) ** 0.5
        z = means + std * eps
        return z

    def call(self, inputs, training=None):
        means, log_var = self.encoder(inputs)   # shape (784,10) => (b,10)
        z = self.reparameterize(means, log_var)  # instead of sampling z, make z = means + std * eps
        x_gen = self.decoder(z)   # shape  (b,10) => (784,10)
        return x_gen, means, log_var


def auto_encoder():
    tf.random.set_seed(8)

    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()  # 28*28
    x_train, x_test = x_train / 255., x_test / 255.
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)   # unsupervised learning, output is x_train
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE,drop_remainder=True)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.batch(BATCH_SIZE,drop_remainder=True)

    '''
    model = AE()
    model.build(input_shape=(None, 784))
    model.summary()
    optimizer = tf.optimizers.Adam(LEARNING_RATE)

    
    for epoch in range(5):
        for step, x in enumerate(train_dataset):
            x = tf.reshape(x, [-1, 784])  # shape  (b,28,28)=> (b, 784)
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = tf.reduce_mean(tf.losses.binary_crossentropy(x, logits, from_logits=True))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

        model = AE()
        model.build(input_shape=(None, 784))
        model.summary()
        optimizer = tf.optimizers.Adam(LEARNING_RATE)
        
    # evaluation
        x = next(iter(test_dataset))
        logits = model(tf.reshape(x, [-1,784]))
        x_hat = tf.sigmoid(logits)
        x_hat = tf.reshape(x_hat,[-1,28,28])   # shape (b, 784) => (b,28,28)

        #x_concat = tf.concat([x, x_hat],axis=0)
        x_concat = x_hat
        x_concat = x_concat.numpy() * 255.
        x_concat = x_concat.astype(np.uint8)
        save_images(x_concat, '../resources/images/ae_%d.png' % epoch)

    '''
    model = VAE()
    model.build(input_shape=(BATCH_SIZE, 784))
    model.summary()
    optimizer = tf.optimizers.Adam(LEARNING_RATE)

    for epoch in range(5):
        for step, x in enumerate(train_dataset):
            x = tf.reshape(x, [-1, 784])  # shape  (b,28,28)=> (b, 784)
            with tf.GradientTape() as tape:
                logits, means, log_var = model(x)
                #gen_loss = tf.reduce_mean(tf.losses.binary_crossentropy(x, logits, from_logits=True))
                #print(logits,x)
                x = tf.cast(x, dtype=tf.float32)
                gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits)
                gen_loss = tf.cast(gen_loss, dtype=tf.float32)
                gen_loss = tf.reduce_sum(gen_loss) / x.shape[0]
                kl_div = -0.5 * (log_var + 1 - means ** 2 - tf.exp(log_var))
                kl_div = tf.reduce_sum(kl_div) / x.shape[0]
                loss = gen_loss + 1. * kl_div
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))
        '''
        # evaluation with test
        x = next(iter(test_dataset))
        logits, _, _ = model(tf.reshape(x, [-1, 784]))
        x_hat = tf.sigmoid(logits)
        x_hat = tf.reshape(x_hat, [-1, 28, 28])  # shape (b, 784) => (b,28,28)

        # x_concat = tf.concat([x, x_hat],axis=0)

        x_hat = x_hat.numpy() * 255.
        x_hat = x_hat.astype(np.uint8)
        save_images(x_hat, '../resources/images/vae_%d.png' % epoch)
        '''

        # generate new data
        z = tf.random.normal((BATCH_SIZE, z_dim))
        logits = model.decoder(z)
        x_gen = tf.sigmoid(logits)
        x_gen = tf.reshape(x_gen, [-1, 28, 28]).numpy() * 255.
        x_gen = x_gen.astype(np.uint8)
        save_images(x_gen, '../resources/images/vae_gen_%d.png' % epoch)





class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # shape (b,3*3*512) => (b, 3, 3, 512)
        self.fc = layers.Dense(3*3*512)
        self.conv1 = layers.Conv2DTranspose(256, 3, 3, 'valid')   # filters, kernel size, stride
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid')


    def call(self, inputs, training=None):
        x = self.fc(inputs)   # shape (z,100) => (z,3*3*512),  z range(-1,1)
        x = tf.reshape(x,[-1, 3, 3, 512])  # shape (z,3*3*512) => (z, 3, 3, 512)
        x = tf.nn.leaky_relu(x)
        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = self.conv3(x)
        x = tf.tanh(x)
        return x

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = layers.Conv2D(64, 5, 3,'valid')  #filters, kernel size, stride
        self.conv2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()

        self.flatten = layers.Flatten()    # (b, h, w, 3) => (b, -1)
        self.fc = layers.Dense(1)


    def call(self, inputs, training=None):
        # shape (b,64,64,3) => (b,1)
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        x = self.flatten(x)
        logits = self.fc(x)

        return logits

def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    toimage(final_image).save(image_path)
    #final_image.save(image_path)



def celoss_ones(logits):

    # calculate label 1 cross entropy loss
    y = tf.ones_like(logits)
    #loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(y, logits=logits)
    return tf.reduce_mean(loss)

    #return 1 - tf.reduce_mean(logits)
def celoss_zeros(logits):

    # calculate label 0 cross entropy loss
    y = tf.zeros_like(logits)
    #loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(y, logits=logits)
    return tf.reduce_mean(loss)


    #return tf.reduce_mean(logits)


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # calculate generator loss loss for DCGAN
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_real = celoss_ones(d_real_logits)  # calculate real image determine as real cross entropy loss
    d_loss_fake = celoss_zeros(d_fake_logits)  # calculate fake image determine as fake cross entropy loss
    loss = d_loss_fake + d_loss_real  # loss for DCGAN combine 2 losses
    return loss

def d_loss_fn2(generator, discriminator, batch_z, batch_x, is_training):
    # calculate generator loss for WGAN
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_real = celoss_ones(d_real_logits)  # calculate real image determine as real cross entropy loss
    d_loss_fake = celoss_zeros(d_fake_logits)  # calculate fake image determine as fake cross entropy loss
    gp = gradient_penalty(discriminator, batch_x, fake_image)
    loss = d_loss_fake + d_loss_real + 1. * gp    # loss for WGAN
    return loss, gp

def g_loss_fn(generator, discriminator, batch_z, is_training):
    # calculate generator loss
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits)  # calculate fake image determine as real cross entropy loss

    return loss

def gradient_penalty(discriminator, batch_x, fake_image):
    # extra term for WGAN
    batchsz = batch_x.shape[0]

    # [b, h, w, c]
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # [b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)

    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate, training=True)
    grads = tape.gradient(d_interplote_logits, interplate)

    # grads:[b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1) #[b]
    gp = tf.reduce_mean( (gp-1)**2 )

    return gp


def GAN():   # DCGAN and WGAN
    '''
    test model correct
    d = Discriminator()
    g = Generator()
    x = tf.random.normal([2, 64,64,3])
    z = tf.random.normal([2,100])
    prob = d(x)
    print(prob)
    x_hat = g(z)
    print(x_hat.shape)
    '''

    tf.random.set_seed(8)
    np.random.seed(8)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    z_dim = 100  # z vector size
    epochs = 300000
    batch_size = 64  # batch size
    learning_rate = 0.0002
    is_training = True

    # get training data
    img_path = glob.glob('../resources/anime/*.jpg') + \
               glob.glob('../resources/anime/*.png')

    print('images num:', len(img_path))

    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size, resize=64)
    print(dataset, img_shape)
    sample = next(iter(dataset))
    print(sample.shape, tf.reduce_max(sample).numpy(),
          tf.reduce_min(sample).numpy())
    dataset = dataset.repeat(100)   # unlimited sampling
    db_iter = iter(dataset)

    generator = Generator()  #
    generator.build(input_shape=(4, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(4, 64, 64, 3))
    # need separate optimizers for generator and discriminator
    g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    if os.path.exists('../resources/generator.ckpt'):
        generator.load_weights('../resources/generator.ckpt')
    if os.path.exists('../resources/discriminator.ckpt'):
        discriminator.load_weights('discriminator.ckpt')


    d_losses, g_losses = [], []
    for epoch in range(epochs):
        # train discriminator
        for _ in range(1):

            batch_z = tf.random.normal([batch_size, z_dim])
            batch_x = next(db_iter)  # sample from real image

            with tf.GradientTape() as tape:
                #d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)  # for DCGAN
                d_loss,gp = d_loss_fn2(generator, discriminator, batch_z, batch_x, is_training) # for WGAN
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # train generator
        batch_z = tf.random.normal([batch_size, z_dim])
        batch_x = next(db_iter)

        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            #print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss))
            print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss), 'gp:', float(gp))
            # generate image sample for visualization
            z = tf.random.normal([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('../resources/gan', 'gan-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')

            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

        if epoch % 10000 == 0:
            generator.save_weights('../resources/generator.ckpt')
            discriminator.save_weights('../resources/discriminator.ckpt')

           







if __name__ == '__main__':
    # example1()
    # mnist_fashion()
    # convolution_pooling()
    # speed()
    # mnist_digits()
    # test()
    # mnist_from_scratch()
    # print(tf.test.gpu_device_name())
    # cifar10()
    # cifar100()
    imdb()
    # auto_encoder()
    #GAN()