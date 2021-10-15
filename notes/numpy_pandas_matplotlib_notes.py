
'''


    numpy  (numeric python)
        numpy smaller and faster than list
        print(np.__version__)
        # create np array (data in array have same data type, will convert automatically,
            #convertion priority: str > float > int)
        a = np.array([[1, 2, 3], [4, 5, 6]])  # np.array([1, 2, 3], dtype='int16')   # array([[1, 2, 3], [4, 5, 6]])
        print(a)  # [[1 2 3][4 5 6]]
        b = np.asarray([[1, 2, 3], [4, 5, 6]])
        b = a  # a and b are two names for the same ndarray object
        c = a.view()  # c is a views of the data owned by a, c resize won't change a, but change entry value will change a
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
        a[::-1,::-1]    # reverse in first and second dimension   array([[6, 5, 4],[3, 2, 1]])
            # plt.imshow(img[:,::-1])    left-right reverse image
        print(np.repeat(a, 2, axis=0))  # [[0 1 0][0 1 0][4 2 0][4 2 0]]
        # doesn't change array unless assign the result, repeat in dimension 0, repeated value beside original
        print(a)  # [[0 1 0][4 2 0]]

        # generate special array
        np.zeros(3)   # [0., 0., 0.]
        np.ones((2, 2))  # 2*2 matrix of 1
        np.full((2, 2), 3, dtype='int16')      # 2*2 matrix of 3, int type
        np.full_like(a, 8))
            # full_like specify shape like one other array
        np.random.rand(3, 2) # even distribution [0, 1)  for 3*2 matrix
        np.random.normal(0, 2, (3, 3))
        np.random.randint(4, 7, size=(2, 2)))  # [4, 7) even distribution;
        print(np.random.randn(2, 4))  # 2*4 array random distribution, mean 0, std 1
        print(np.arange(0, 3, 1))  # [0,3) step 1  [0, 1, 2], not consistent with decimal, use linspace
        print(np.linspace(2.0, 3.0, num=5))  # [2,3] divide 5 points 1  [2.   2.25 2.5  2.75 3.  ]
            # add endpoint=False, not include 3,  # retstep return a tuple with list and step size
        print(np.identity(2))   print(np.eye(2))  # identity matrix [[1. 0.][0. 1.]]
        np.eye(2, k=1)  # daigpnal shift up 1 position

        # aggregate functions
        print(np.min(a), np.min(a, axis=1), np.sum(a, axis=0)) or a.sum(axis=0)   # 0 [0 0] [4 3 0]
            prod,  mean, std, var, argmin, argmax, median, precentile, any, all, power
            np.nan  # empty value (not a number)
            np.nansum(a)  # don't consider np.nan in sum

        # other operations
        print(a.ravel()[1::2])  # flatten to 1d array  [0 1 0 4 2 0] then from 1 step 2 [1 4 0]
        print(a.reshape(3, 2))  # [[0 1][0 4][2 0]]   size must same
        # a.resize((3, 2)) will modify original shape
        np.split(n,[2,4],axis=0)   split n into array of 3 part: line 0,1; line 2,3; line 4-end
        print(np.vstack([np.ones(3), np.zeros((2, 3))]))  # [[1 1 1][0 0 0][0 0 0]]
        print(np.hstack([np.ones((2, 1)), np.zeros((2, 2))]))  # [[1. 0. 0.][1. 0. 0.]]
        print(np.hsplit(a, 3))  # [array([[0],[4]]), array([[1],[2]]), array([[0],[0]])]
            np.hsplit(n, [2,4])  same as  np.split(n,[2,4],axis=0)
        print(np.vsplit(a, 2))  # [array([[0, 1, 0]]), array([[4, 2, 0]])]
        print(np.concatenate((np.ones((1, 3)), np.zeros((2, 3))), axis=0))  # [[1. 1. 1.][0. 0. 0.][0. 0. 0.]]
        print(np.concatenate((np.ones((1, 3)), np.zeros((2, 3))), axis=None))  # [1. 1. 1. 0. 0. 0. 0. 0. 0.]
        np.concatenate((np.ones((2, 2)), np.zeros((2, 3))), axis=1)  #[[1., 1., 0., 0., 0.],[1., 1., 0., 0., 0.]]
            # final result and original same size in axis dimension

        # entry value modification
        print(a * 2, a + 2, a ** 2, a * a, np.sin(a), np.sqrt(a), np.exp(a), np.log([1, 2]))
        # doesn't change array unless assign, each value times 2, sin(), log()
        a *= 1  # this will update original matrix
        # broadcast rule, add dimension for smaller ndarray, fill added dimension with existing dimension value
        x,y = np.ones((2,3)), np.full((3), 3)
        x+y   # [[4,4,4][4,4,4]]  y become [[3,3,3][3,3,3]]


        # matrix modification
        print(np.matmul(a, a.transpose()))  # matrix multiplication  same as a.T
        print("x", np.dot(a, a.transpose()))  # matrix product same as a @ a.transpose()
        # For 2-D arrays it is equivalent to matrix multiplication, and for 1-D arrays to inner product of vectors (without
        # complex conjugation). For N dimensions it is a sum product over the last axis of a and the second-to-last of b
        a*b    # same size matrix a and b same position multiplication, return same size matrix
        print(np.inner([1, 2, 3], [0, 1, 4]))  # 14  1*0+2*1+3*4  (1*3)*(3*1)=(1*1)
        # inner product: sum of multiplication of a and b same position element, b's projection on a
        print(np.cross([1, 2, 3], [0, 1, 4]))
        # [2*4-3*1, -1*4+3*0, 1*1-2*0]  [ 5 -4  1] cross product, direction that perpendicular to a and b
        print("z", np.outer([1, 2, 3], [0, 1, 4]))  # [[ 0  1  4][ 0  2  8][ 0  3 12]]  outer product of a and b is ab^T
        print(np.dot(np.array([1, 2, 3]).reshape(3, 1), [[0, 1, 4]]))  # [[ 0  1  4][ 0  2  8][ 0  3 12]]  (3*1)*(1*3)=(3*3)
            np.dot(a,b) != np.dot(b,a)
        print(np.linalg.det([[1, 2], [3, 4]]))  # -2  input can be list or np.array
        # np.linalg.svd (Singular Value Decomposition)
        print(np.linalg.eigvals([[1, 2], [3, 4]]))  # square matrix [-0.37228132  5.37228132]

        # Compute the eigenvalues of a general matrix.

        # read txt file
        file_data = np.genfromtxt(os.path.join('..','resources','data','numpy_data.txt'), delimiter=',').astype('int32')
            # default float
        print(file_data)  # [[ -1   1  13 196   0][  3  42  12  33 766][  1  22  33  11 999]]
        print(file_data[file_data > 50])  # [196 766 999]
        print((~((file_data > 0) & (file_data < 100))))  # s<=0 or >=100
        # [[ True False False  True  True][False False False False  True][False False False False  True]]
        print(np.any([[True, False], [False, False]], axis=0))  # [ True False]
        print(np.all([[True, False], [False, False]], axis=0))  # [False False]
        print(np.where(a < 1, a, -1))  # if element < 1, return element, otherwise -1 [[ 0 -1  0][-1 -1  0]]


    pandas
        2 main datatypes: series, dataframe
        Series: 1 dimension
            import pandas as pd
            create series: can be created using np.array or list or dictionary
            brand = pd.Series(["BMW", "Toyota", "Honda"])    #takes in list
            color = pd.Series(["Red", "Blue", "White"], index=[0,3,9])  # assign custom index
            color = pd.Series(np.array(["Red", "Blue", "White"]), index=list('abc'))  # assign custom index
            color = pd.Series({"a":"Red", 'b':"Blue", 'c':"White"})

            color['a'] = "Green"  # update    need pd.Series(arr.copy()) if arr is np.array with number, otherwise
                update will change arr value as well
            explicit index
                color.loc['a']   #retrieve value,
                    color['a'] # not recommended, can't distinguish explicit or implicit indexing
                color.loc['a':'c']  # index ['a','c']
            implicit index
                color.iloc[0]
                color.iloc[0:3]   # index [0,3)
                color[0]  # not recommended, can't distinguish explicit or implicit indexing

            attribute
            color.shape  # (3,)         color.size  # 3
            color.index   Index(['a', 'b', 'c'], dtype='object')
            color.values # array(['Red', 'Blue', 'White'])
            color.head()    # first 5 elements ,  or head(8)  first 8 elements
            color.name = 'color'   # assign name for series

            pd.isnull(color)  # return index and boolean of null check for np.NaN
                pd.notnull(color)   color.isnull(s)   color.notnull()

            math operation: don't have broadcast, fill NaN if missing (number add NaN is NaN)
            +, -, *, /  apply to all number elements
            a + b (same explicit index base operation ex.a.loc[1]+b.loc[1]..., if one is NaN, return NaN)
            a.add(b, fill_value=0)    # instead of fill NaN, fill 0
                subtract()/sub()    multiply()/mul()   divide()/div()    floordiv()   mod()  pow()
            aggregate function same as numpy a.sum()

        DataFrame: 2 dimensional
            car_data = pd.DataFrame({"Car make": brand, "Color": color})  #takes in dictionary or 2 series
            car_sales = pd.read_csv("car-sales.csv")   #import from structure data csv
            axis = 0: row      axis = 1:column
            car_sales = pd.read_csv("car-sales.csv", parse_date=["sale_date"])   #convert to date object
            df["Year"] = df.saledate.dt.year

            car_sales.to_csv("new-sales.csv", index=False)  # export to csv file, excluding index column
        Functions (with()) and Attribute:
            .dtypes   #show column datatype
            .columns   # return list of column name
            .index    # return  rangeIndex object contain start, end, step


            .describe()   # return statistic information of numerical columns
            .info()    # information of index + dtypes
            .mean()     # return mean of numerical columns
            .sum()    # sum of numerical columns, concatenate object column
            len(car_sales)   # return rows count
            .head()     # return default show 5 rows of data
                .head().T   # if too many columns and truncated
            .tail()     # last 5 rows
            .sort_values(by=["saledate"], inplacce=True, ascending=True)

            .loc[3]   # return item with index 3
            car_sales.loc[1, "price"] =3000   # set value for a cell
            .iloc[3]  # return item at row 4    .iloc[:3]   # return row 1-4
            car_sales["Doors"].plot()  # return doors column in plot
                or  car_sales.Doors can't have space in name
                %matplotlib inline   import matplotlib.pyplot as plt   if plot not show up
            car_sales[car_sales["Doors"]  == 4]   # return cars with 4 doors
            pd.crosstab(car_sales["Make"], car_sales["Doors"] )    #return table of 2 selected columns as x
                and y
            .groupby(["Make"]).mean()["Price"]   #return mean price for each make
            .plot()   .hist() # histogram
            .value_counts()  # return count of series unique item

            car_sales["Make"] = car_sales["Make"].str.lower()   #change make column to lower case,
                need reassign if any change of column
            car_sales["Doors"].fillna(car_sales["Doors"].mean(), inplace=True)    # fill missing value cell
                replacing NaN with average value, inplace don't need reassign, change in place
            car_sales = car_sales.dropna()   #remove row with nan
            car_sales.dropna(subset=["Target"], inplace=True)   # remove row with target is na

            Add/remove new column
            car_sales["Seats"] = pd.Series([5,4,5,6])   #add a seat column with 4 rows
                car_sales["Seats"]  = [5,4,5,6,9,10]     # need have same rows as original table
                car_sales["Seats"] = 4   # all 4 seats
            car_sales = car_sales.drop("Seats", axis = 1)   # drop column seats


            car_sales = car_sales.sample(frac=0.6)    # shuffle and use 60% data, index same, row change
            car_sales = car_sales.reset_index(drop=True)  # reset the index from 0 step 1, don't add new
                index
            car_sales["Price"] = car_sales["Price"].apply(lambda x: x*6.5)  # each price * 6.5, apply
                function to change column value
            car_sales["Price"] = car_sales["Price"].astype(int)  pd.to_numeric(car_sales["Price"])
            zip
        # change string column to category
        for label, value in df.items():
            if pd.api.types.is_string_dtype(value):
                df["label"] = value.astype("category").cat.as_ordered()   # change string column to
                    category order by alphabet. change to int underneath,
                df.state.cat.codes  # show int code of category column "state"
            pd.Categorical(df["state"])  #list of states (categorical)
        #check null
        for label, value in df.items():
            if pd.api.types.is_numeric_dtype(value):
                if pd.isnull(value).sum():
                    df[lanel+"_is_misiing"] = pd.isnull(value)
                    df[label] = value.fillna(value.median())
            if not pd.api.types.is_numeric_dtype(value):
                df[lanel+"_is_misiing"] = pd.isnull(value)
                df[label] = pd.Categorical(value).codes + 1  #default na category code is -1



    matplotlib
        %matplotlib inline     # show plot/figure in console
        # plot image with matplotlib
        img = plt.imread('./resources/images/hp2.jpg')   # return ndarray (matrix) of image.  need pillow if not png file
            # png [0,1] h*w*4    jpg: [0,255] h*w*3
        plt.imshow(img)
        plt.imshow(img[:,::-1])  # left right reverse
        plt.imshow(img[::2,::2])  # compress 4 pixcel become 1


'''
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def numpy_basic():
    # create np array
    a = np.array([[1, 2, 3], [4, 5, 6]])  # np.array([1, 2, 3], dtype='int16')
    print(a)  # [[1 2 3][4 5 6]]
    b = np.asarray([[1, 2, 3], [4, 5, 6]])
    b = a  # a and b are two names for the same ndarray object
    c = a.view()  # c is a views of the data owned by a, c resize won't change a, but change entry value will change a
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
    file_data = np.genfromtxt(os.path.join('..','resources','data','numpy_data.txt'), delimiter=',').astype('int32')
        # default float
    print(file_data)  # [[ -1   1  13 196   0][  3  42  12  33 766][  1  22  33  11 999]]
    print(file_data[file_data > 50])  # [196 766 999]
    print((~((file_data > 0) & (file_data < 100))))  # s<=0 or >=100
    # [[ True False False  True  True][False False False False  True][False False False False  True]]
    print(np.any([[True, False], [False, False]], axis=0))  # [ True False]
    print(np.all([[True, False], [False, False]], axis=0))  # [False False]
    print(np.where(a < 1, a, -1))  # if element < 1, return element, otherwise -1 [[ 0 -1  0][-1 -1  0]]


if __name__ == '__main__':
   numpy_basic()