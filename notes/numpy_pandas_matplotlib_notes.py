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
            np.nan  # empty value (not a number, float type, able to calculate, but result is nan)
                # pandas will convert None to np.nan
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
            # create series: can be created using np.array or list or dictionary
            brand = pd.Series(["BMW", "Toyota", "Honda"])    #takes in list
            color = pd.Series(["Red", "Blue", "White"], index=[0,3,9])  # assign custom index
            color = pd.Series(np.array(["Red", "Blue", "White"]), index=list('abc'))  # assign custom index
            color = pd.Series({"a":"Red", 'b':"Blue", 'c':"White"})

            # hierarchical index
            color = pd.Series(["Red", "Blue", "White"], index=[['Light','Light','Dark'],[0,3,9]])


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

            # add value
            color['d'] = np.nan
            color = color.append(pd.Series(['Orange','Black']))  # default index start at 0, {"e":'Orange', "f":'Black'}

            # attribute
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
            aggregate function same as numpy a.sum()  # add numeric, concatenate string
                prod,  mean, std, var, argmin, argmax, median, precentile, any, all, power

        DataFrame: 2 dimensional
            each row or column is a Series

            # create dataframe
            df = pd.DataFrame(data=np.random.randint(0, 100, size=(2, 2)), index=['Magic Defense', 'Magic Spell'],
                columns=['Harry','Ronald'])   # data: 2D list/np.array, index:list, column:list
            df = pd.DataFrame({'Harry': np.random.randint(0, 100, size=2), 'Ronald': np.random.randint(0, 100, size=2)},
                index=['Magic Defense','Magic Spell'])   # dictionary: key:column name, value:list;  index:list
            df = pd.DataFrame({'Harry': np.random.randint(0, 100, size=2), 'Ronald': np.random.randint(0, 100, size=2)})
            df.index = ['Magic Defense','Magic Spell']  # assign later


            car_sales = pd.read_csv("car-sales.csv")   #import from structure data csv
            axis = 0: row      axis = 1:column
            car_sales = pd.read_csv("car-sales.csv", parse_date=["sale_date"])   #convert to date object
            df["Year"] = df.saledate.dt.year
            car_sales.to_csv("new-sales.csv", index=False)  # export to csv file, excluding index column

            # access column/columns via column index
            df['Harry']   # Magic Defense    81
                            Magic Spell      32
                            Name: Harry, dtype: int32
                # or df.Harry   # not recommended, can't have space in column name
            df[['Harry','Hermione']]   # select columns
            df.iloc[:, 0:2]   # select columns  iloc  [0,2)

            # access row/rows via row index
            df.loc['Magic Defense']  # explicit index
            df.iloc[0]  # implicit index
            df.loc['Magic Defense':'Magic Spell']   # ['Magic Defense','Magic Spell']  inclusive
            df2.iloc[0:1]  # rows [0, 1)

            # add/remove column
            df['Hermione'] = [99,98]     # pd.Series([99,98])
            df['Hermione'] = 100   # 100 for all rows
            df = df.drop("Hermione", axis = 1)   # drop column Hermione

            #add row
            df = df.append(pd.DataFrame({'Harry':88, 'Ronald':60, ''Hermione':97}, index=['Magic Creature']))
            df.loc['Magic Creature'] = pd.Series({'Harry': 88, 'Ronald': 60, 'Hermione': 97})

            # access data cell
            df.loc['Magic Defense', 'Hermione']  # recommended, first row then column
                # df.loc['Magic Defense', 'Hermione'] = 100
                # df2.iloc[0,2] = 98  # iloc first row then column
            # not recommended
                df['Hermione'].loc['Magic Defense'] # first column then row, chain index might cause issue during update
                df['Hermione']['Magic Defense']
                df.loc['Magic Defense'].loc['Hermione']  # first row then column


            # hierarchical index
            df = pd.DataFrame(data=np.random.randint(0, 100, size=(2, 2)), index=[['Grade1', 'Grade2'], ['Magic Defense'
                ,'Magic Spell']],columns=[['Male','Male']['Harry','Ronald']])
            # 3 ways: pd.MultiIndex.from_arrays, from_tuples, product
            df = pd.DataFrame(data=np.random.randint(0, 100, size=(3, 3)), index=pd.MultiIndex.from_arrays([['Grade1',
                'Grade1','Grade3'], ['Magic Defense','Magic Spell','Magic Creature']]),columns=pd.MultiIndex.from_tuples
                ([('Male','Harry'),('Male','Ronald'),('Female','Hermione')]))
                # index=pd.MultiIndex.from_product([[Grade1, Grade2], ['Magic Defense','Magic Spell']])
                    # Cartesian product, all combination

            df.loc['Grade1']  # all rows with Grade1
            df.loc['Grade1','Magic Defense']   # row of Magic Defense
            df.iloc[[0]]  # first row as a dataframe
            df.loc[('Grade1','Magic Defense'):('Grade1','Magic Spell')]  # slice first 2 rows
            df.iloc[0:2]  # slice first 2 rows
            df.loc['Grade1','Magic Defense']['Male','Harry']   # cell Magic Defense, Harry
                # df.loc[('Grade1','Magic Defense'),('Male','Harry')]
            df.iloc[0:2,0]  # [0,2) row, first item


            df['Male','Harry']  # Harry column
            df.loc[:, ('Male', ['Harry','Ronald'])]   # slice first 2 column
            df.iloc[:, 0:2]   # slice [0,2) column

            df.stack()  # column become 'Male','Female', convert inner column index 'Harry','Ronald','Hermione' to most
                inner row index (total 3 layers row index), can generate additional  np.nan
                # df.stack(level=0, fill_value=0)  # convert level 0 index ('Male','Female') to column index
            df.unstack()  # row become 'Grade1','Grade1','Grade2', convert 'Magic Defense','Magic Spell','Magic
                Creature'  to most inner column index (total 3 layers column index)


            #Functions (with()) and Attribute:
            .dtypes    # show column datatype
            .columns   # return list of column name
            .index     # Index(['Magic Defense', 'Magic Spell'], dtype='object')
            .values    # data inside table (2d np.array)
            .shape     # values (data) shape (2,2)

            .describe()   # return statistic information of numerical columns
            .info()    # information of index + dtypes
            len(car_sales)   # return rows count
            .head()     # return default show 5 rows of data
                .head().T   # if too many columns and truncated
            .tail()     # last 5 rows
            .sort_values(by=["saledate"], inplacce=True, ascending=True)
            df = df.astype(dtype=np.int16)  or df['Harry'] = df['Harry'].astype(int)  # change datatype

            aggregate function  # add numeric, concatenate string, return dataframe
                df.sum()  #  for each column sum a value, return dataframe
                df.sum(axis=1)  # each row sum a value(same column)
                df.sum(axis=1,level=0)  # each level 0 row index sum a value (same column)
                prod,  mean, std, var, argmin, argmax, median, precentile, any, all, power

            car_sales["Harry"].plot()  # return doors column in plot
            car_sales[car_sales["Doors"]  == 4]   # return cars with 4 doors
            pd.crosstab(car_sales["Make"], car_sales["Doors"] )    #return table of 2 selected columns as x
                and y
            .groupby(["Make"]).mean()["Price"]   #return mean price for each make
            .plot()   .hist() # histogram
            .value_counts()  # return count of series unique item


            pd.isnull(df)    pd.notnull()    df.isnull()   df.notnull()
            df.isnull().any()  # return each column boolean whether has nan, default axis=0
            df = df.dropna()  # delete roll with na, default axis=0, how='any'    (how='all': drop row full nan)
                # same as df.dropna(inplace = True)  # default inplace = False, need assign
                # df.dropna(subset=['Harry','Ronald'])   only drop for nan in certain column
            df.fillna(value=100)   # fill nan with 100
            df = df.fillna(axis=0, method='bfill')  # fill nan with value below. backfill, pad(fill with left), ffill
                # limit=2   max fill 2 consecutive nan
            df['Hermione'].fillna(df['Hermione'].mean(), inplace=True)

            car_sales["Make"] = car_sales["Make"].str.lower()   #change make column to lower case,
                need reassign if any change of column
            car_sales = car_sales.sample(frac=0.6)    # shuffle and use 60% data, index same, row change
            car_sales = car_sales.reset_index(drop=True)  # reset the index from 0 step 1, don't add new
                index
            car_sales["Price"] = car_sales["Price"].apply(lambda x: x*6.5)  # each price * 6.5, apply
                function to change column value
            car_sales["Price"] = car_sales["Price"].astype(int)  pd.to_numeric(car_sales["Price"])
            zip
            # change string column to category, default nan category code is -1
            for label, value in df.items():
                if pd.api.types.is_string_dtype(value):
                    df["label"] = value.astype("category").cat.as_ordered()   # change string column to
                        category order by alphabet. change to int underneath,
                    df.state.cat.codes  # show int code of category column "state"
                pd.Categorical(df["state"])  #list of states (categorical)


            # math operation
            don't have broadcast, fill NaN if missing (number add NaN is NaN)
            python None can't used to calculate, pandas convert None to np.nan. np.nan math operation always return nan
            +, -, *, /  apply to all number elements
            a + b (same explicit index base operation ex.a.loc[0,0]+b.loc[0,0]..., if one is NaN, return NaN)
            a.add(b, fill_value=0)    # instead of fill NaN, fill 0, need both dataframe, only same type can use
                subtract()/sub()    multiply()/mul()   divide()/div()    floordiv()   mod()  pow()
            # data frame and series operation default based on column index and series index, use add() and axis='index'
                to switch to dataframe row index and series index
            # dataframe and series operation based on df column index and series index
            delta = pd.Series([10, -2], index = ['Harry','Hermione'])
            print(df2 + delta) # add 10 for Harry column and minus 2 for Hermione column, missing column fill NaN
                or df2.iloc[:,:] = df2.iloc[:,:]-[10,0,-2]
            # dataframe and series operation based on df row index and series index
            delta2 = pd.Series([0, 0, -10], index=['Magic Defense', 'Magic Spell', 'Magic Creature'])
            print(df2.add(delta2, axis='index'))  # minus 10 for Magic Creature row
                or df2.iloc[2:, :] = df2.iloc[2:, :] - 10  # df+number operation for each item

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
    file_data = np.genfromtxt(os.path.join('..', 'resources', 'data', 'numpy_data.txt'), delimiter=',').astype('int32')
    # default float
    print(file_data)  # [[ -1   1  13 196   0][  3  42  12  33 766][  1  22  33  11 999]]
    print(file_data[file_data > 50])  # [196 766 999]
    print((~((file_data > 0) & (file_data < 100))))  # s<=0 or >=100
    # [[ True False False  True  True][False False False False  True][False False False False  True]]
    print(np.any([[True, False], [False, False]], axis=0))  # [ True False]
    print(np.all([[True, False], [False, False]], axis=0))  # [False False]
    print(np.where(a < 1, a, -1))  # if element < 1, return element, otherwise -1 [[ 0 -1  0][-1 -1  0]]


def pandas_basic():
    # Series
    brand = pd.Series(["BMW", "Toyota", "Honda"])  # create by list
    color = pd.Series(np.array(["Red", "Blue", "White"]), index=list('abc'))  # create by np.array assign custom index
    color_array = np.array(["Red", "Blue", "White"])
    color = pd.Series(color_array.copy(), index=[0, 3, 9])  # assign custom index
    color = pd.Series({"a": "Red", 'b': "Blue", 'c': "White"})  # create by dictionary
    color['a'] = "Green"  # update    need pd.Series(arr.copy()) if arr is np.array with number, otherwise update will
        # change np.array value as well
    # explicit index
    print(color.loc['a'])  # retrieve value,
        # print(color['a']) # not recommended, can't distinguish explicit or implicit indexing
    print(color.loc['a':'c'])  # index ['a','c']
    # implicit index
    print(color.iloc[0])
    print(color.iloc[0:2])  # index [0,2)
        # print(color[0])  # not recommended, can't distinguish explicit or implicit indexing
    print(color.shape)  # (3,)         color.size  # 3
    print(color.index)   # Index(['a', 'b', 'c'], dtype='object')
    print(color.values)   # array(['Red', 'Blue', 'White'])
    print(color.head(), color.head(2))   # first 5 elements ,  or head(8)  first 8 elements
    color.name = 'color'  # assign name for series

    color['d'] = np.nan
    color = color.append(pd.Series(['Orange','Black']))  # {"e":'Orange', "f":'Black'}   default index start at 0
    print(color)  # a Green, b Blue, c White, d NaN, 0 Orange, 1 Black
    print(pd.isnull(color))  # return index and boolean of null check for np.NaN
        #pd.notnull(color)    color.isnull()  color.notnull()
    a = pd.Series(np.arange(0,3))   # [0,3)
    b = pd.Series(np.arange(1,4), index=[1,2,3])
    print(a+b)  # 0 NaN    1 2.0       2 4.0      3 NaN
        # same explicit index base operation ex.a.loc[1]+b.loc[1]..., if one is NaN, return NaN
    print(a.add(b, fill_value=0))   # 0 0.0    1 2.0       2 4.0      3 3.0
        # instead of fill NaN, fill 0  subtract()/sub(), multiply()/mul(), divide()/div(), floordiv(), mod(), pow()



    # DataFrame
    df = pd.DataFrame(data=[[1,2],[3,4]], index=['Magic Defense', 'Magic Spell'],
                                             columns=['Harry','Ronald'])  # data: list/np.array, index:list, column:list
    df2 = pd.DataFrame({'Harry': np.random.randint(0, 100, size=2), 'Ronald': np.random.randint(0, 100, size=2)}, index=
        ['Magic Defense','Magic Spell'])   # dictionary with column as key and list of value as value, index: list
    df2['Hermione'] = [99, 99]
    #df2 = df2.append(pd.DataFrame({'Harry': 88, 'Ronald': 60, 'Hermione':97}, index=['Magic Creature']))
    df2.loc['Magic Creature'] = pd.Series({'Harry': 88, 'Ronald': 60, 'Hermione': 97})
    df2.loc['Magic Defense', 'Hermione'] = 100
    df2.iloc[0,2] = 98
    print(df2)
    print(df2.iloc[:, 0:2])
    print(df2[['Harry','Hermione']] )
    delta = pd.Series([10, 0, -2], index = ['Harry','Ronald', 'Hermione'])
        # add 10 for Harry column and minus 2 for Hermione column
    delta2 = pd.Series([0, 0, -10], index=['Magic Defense', 'Magic Spell', 'Magic Creature'])
        # minus 10 for Magic Creature row
    print(df2.add(delta))
    print(df2.add(delta2, axis='index'))
    print(df2)
    #df2.iloc[:,:] = df2.iloc[:,:]-[0,5,10]
    df2.iloc[1:, :] = df2.iloc[1:, :] - 10
    print(df2)
    df2.iloc[2,2] =np.nan

    df = pd.DataFrame(data=np.random.randint(0, 100, size=(3, 3)), index=pd.MultiIndex.from_arrays([['Grade1','Grade1',
        'Grade3'],['Magic Defense','Magic Spell','Magic Creature']]),columns=pd.MultiIndex.from_tuples([('Male',
        'Harry'), ('Male','Ronald'),('Female','Hermione')]))
    print(df)
    print(df.iloc[0:2])  # slice first 2 rows
    print(df['Male','Harry'])
    print(df.loc[:, ('Male', ['Harry','Ronald'])])
    print(df.loc['Grade1'])
    print(df.loc[('Grade1','Magic Defense'):('Grade1','Magic Spell')])
    print(df.stack())
    print(df.unstack())
    print('x',df.sum().iloc[0:2])


if __name__ == '__main__':
    # numpy_basic()
    pandas_basic()
