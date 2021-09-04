import numpy as np

if __name__ == '__main__':
    # create np array
    a = np.array([[1, 2, 3], [4, 5, 6]])  # np.array([1, 2, 3], dtype='int16')
    print(a)  # [[1 2 3][4 5 6]]
    b = np.asarray([[1, 2, 3], [4, 5, 6]])
    b = a  # a and b are two names for the same ndarray object
    c = a.view()  # c is a view of the data owned by a, c resize won't change a, but change entry value will change a
    d = a.copy()  # d doesn't share anything with a

    # asarray like array, except it has fewer options, and copy=False. array has copy=True by default.

    # common attributes
    print(a.ndim, a.shape, a.dtype, a.itemsize, a.size, a.nbytes)  # 2  (2, 3)  int32  4  6  24
    # dimension,  shape,  data type, each entry data space in byte, items count, total bytes

    # access entry
    print(a[1, -1], a[0, :], a[1, 0:-1:2])  # 6 [1 2 3]  [4]    first item, step size 2, not inclusive last item
    # update entry
    a[0, 0] = 0
    a[:, 2] = 0  # update column, can update to same value or specific values in list
    a[:, 1] = [1, 2]
    print(np.repeat(a, 2, axis=0))  # [[0 1 0][0 1 0][4 2 0][4 2 0]]
    # doesn't change array unless assign the result, repeat in dimension 0, repeated value beside original
    print(a)  # [[0 1 0][4 2 0]]

    # generate special array
    print(np.zeros(3), np.ones((2, 2)), np.full((2, 2), 3, dtype='int16'), np.full_like(a, 8))
    # full_like specify shape like one other array
    print(np.random.rand(3, 2), np.random.normal(0, 2, (3, 3)), np.random.randint(4, 7, size=(2, 2)))
    # rand generate [0, 1) even distribution;  mean 2, std 2, normal distribution; [4, 7) even distribution;
    print(np.arange(0, 3, 1))  # [0,3) step 1  [0, 1, 2]
    print(np.linspace(2.0, 3.0, num=5))  # [2,3] divide 5 points 1  [2.   2.25 2.5  2.75 3.  ]
    print(np.identity(2))  # identity matrix [[1. 0.][0. 1.]]

    # aggregate functions
    print(np.min(a), np.min(a, axis=1), np.sum(a, axis=0))  # 0 [0 0] [4 3 0]

    # other operations
    print(a.ravel()[1::2])  # flatten to 1d array  [0 1 0 4 2 0] then from 1 step 2 [1 4 0]
    print(a.reshape(3, 2))  # [[0 1][0 4][2 0]]
    # a.resize((3, 2)) will modify original shape
    print(np.vstack([np.ones(3), np.zeros((2, 3))]))  # [[1 1 1][0 0 0][0 0 0]]
    print(np.hstack([np.ones((2, 1)), np.zeros((2, 2))]))  # [[1. 0. 0.][1. 0. 0.]]
    print(np.hsplit(a, 3))  # [array([[0],[4]]), array([[1],[2]]), array([[0],[0]])]
    print(np.vsplit(a, 2))  # [array([[0, 1, 0]]), array([[4, 2, 0]])]
    print(np.concatenate((np.ones((1, 3)), np.zeros((2, 3))), axis=0))  # [[1. 1. 1.][0. 0. 0.][0. 0. 0.]]
    print(np.concatenate((np.ones((1, 3)), np.zeros((2, 3))), axis=None))  # [1. 1. 1. 0. 0. 0. 0. 0. 0.]

    # entry value modification
    print(a * 2, a + 2, a ** 2, a * a, np.sin(a), np.sqrt(a), np.exp(a), np.log([1, 2]))
    # doesn't change array unless assign, each value times 2, sin(), log()
    a *= 1  # this will update original matrix

    # matrix modification
    print(np.matmul(a, a.transpose()))  # matrix multiplication  same as a.T
    print("x", np.dot(a, a.transpose()))  # matrix product same as a @ a.transpose()
    # For 2-D arrays it is equivalent to matrix multiplication, and for 1-D arrays to inner product of vectors (without
    # complex conjugation). For N dimensions it is a sum product over the last axis of a and the second-to-last of b
    print(np.inner([1, 2, 3], [0, 1, 4]))  # 14  1*0+2*1+3*4  (1*3)*(3*1)=(1*1)
    # inner product: sum of multiplication of a and b same position element, b's projection on a
    print(np.cross([1, 2, 3], [0, 1, 4]))
    # [2*4-3*1, -1*4+3*0, 1*1-2*0]  [ 5 -4  1] cross product, direction that perpendicular to a and b
    print("z", np.outer([1, 2, 3], [0, 1, 4]))  # [[ 0  1  4][ 0  2  8][ 0  3 12]]  outer product of a and b is ab^T
    print(np.dot(np.array([1, 2, 3]).reshape(3, 1), [[0, 1, 4]]))  # [[ 0  1  4][ 0  2  8][ 0  3 12]]  (3*1)*(1*3)=(3*3)
    print(np.linalg.det([[1, 2], [3, 4]]))  # -2  input can be list or np.array
    # np.linalg.svd (Singular Value Decomposition)
    print(np.linalg.eigvals([[1, 2], [3, 4]]))  # square matrix [-0.37228132  5.37228132]

    # Compute the eigenvalues of a general matrix.

    # read txt file
    file_data = np.genfromtxt('numpy_data.txt', delimiter=',').astype('int32')  # default float
    print(file_data)  # [[ -1   1  13 196   0][  3  42  12  33 766][  1  22  33  11 999]]
    print(file_data[file_data > 50])  # [196 766 999]
    print((~((file_data > 0) & (file_data < 100))))  # s<=0 or >=100
    # [[ True False False  True  True][False False False False  True][False False False False  True]]
    print(np.any([[True, False], [False, False]], axis=0))  # [ True False]
    print(np.all([[True, False], [False, False]], axis=0))  # [False False]
    print(np.where(a < 1, a, -1))  # if element < 1, return element, otherwise -1 [[ 0 -1  0][-1 -1  0]]
