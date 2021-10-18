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
        np.random.permutation([0,1,2,3])  # return a random permutation of array with all elements

        print(np.arange(3))  # default step 1, start 0
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
        np.unique([1,1,2,3])  # array([1,2,3]) return np.array with unique value
        np.histogram(np.array([1,1,2,3,5]))

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
            brand = pd.Series(["BMW", "Toyota", "Honda"])    #takes in list, data=["BMW", "Toyota", "Honda"]
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

            color.unique()   # remove duplicate value
            color.plot()   # kind='bar',  'hist' (bins=10, density=True), 'kde'  #kernel density estimate(install scipy)
                plt.show()  #needed for pycharm to show figure
                # same as dataframe, check below

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
            df = pd.DataFrame([[0,1],[2.3]])  # only have data, default index 0,1,2...
            df = pd.DataFrame(data=np.random.randint(0, 100, size=(2, 2)), index=['Magic Defense', 'Magic Spell'],
                columns=['Harry','Ronald'])   # data: 2D list/np.array, index:list, column:list
            df = pd.DataFrame({'Harry': np.random.randint(0, 100, size=2), 'Ronald': np.random.randint(0, 100, size=2)},
                index=['Magic Defense','Magic Spell'])   # dictionary: key:column name, value:list;  index:list
            df = pd.DataFrame({'Harry': np.random.randint(0, 100, size=2), 'Ronald': np.random.randint(0, 100, size=2)})
            df.index = ['Magic Defense','Magic Spell']  # assign later

            # access column/columns via column index
            df['Harry']   # Magic Defense   81  # get single column (Series type), add additional [] change to DataFrame
                            Magic Spell     32
                            Name: Harry, dtype: int32
                # or df.Harry   # not recommended, can't have space in column name
            df[['Harry','Hermione']]   # select multiple columns and put into dataframe
            df.iloc[:, 0:2]   # select columns  iloc  [0,2)

            # access row/rows via row index
            df.loc['Magic Defense']  # explicit index
            df.iloc[0]  # implicit index
            df.loc['Magic Defense':'Magic Spell']   # ['Magic Defense','Magic Spell']  inclusive
            df2.iloc[0:1]  # rows [0, 1)

            # add/remove column
            df['Hermione'] = [99,98]     # pd.Series([99,98])
            df['Hermione'] = 100   # 100 for all rows
            df = df.drop("Hermione")   # drop column Hermione

            #add row
            df = df.append(pd.DataFrame({'Harry':88, 'Ronald':60, ''Hermione':97}, index=['Magic Creature']))
                # verify_integrity=True   #raise error if has duplicate index after apend
                # ignore_index=True    # reassign index      sort=True
            df.loc['Magic Creature'] = pd.Series({'Harry': 88, 'Ronald': 60, 'Hermione': 97})
            df = df.drop('Magic Creature')   # drop row 'Magic Creature'

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


            # Nan
            pd.isnull(df)    pd.notnull()    df.isnull()   df.notnull()
            df.isnull().any()  # return each column boolean whether has nan, default axis=0
                df.isnull().any(axis=1)  # return each row boolean whether has nan
                df[df.isnull().any(axis=1)]  # return all rows with no nan
            df = df.dropna()  # delete roll with na, default axis=0, how='any'    (how='all': drop row full nan)
                # same as df.dropna(inplace = True)  # default inplace = False, need assign
                # df.dropna(subset=['Harry','Ronald'])   only drop for nan in certain column
            df.fillna(value=100)   # fill nan with 100
            df = df.fillna(axis=0, method='bfill')  # fill nan with value below. backfill, pad(fill with left), ffill
                # limit=2   max fill 2 consecutive nan
            df['Hermione'].fillna(df['Hermione'].mean(), inplace=True)


            #Functions (with()) and Attribute:
            df.dtypes    # show column datatype
            .columns   # return list of column name
            .index     # Index(['Magic Defense', 'Magic Spell'], dtype='object')
            .values    # data inside table (2d np.array)
            .shape     # values (data) shape (2,2)

            .describe()   # return statistic information of numerical columns
            .info()    # information of index count, column name, data type, size
            len(car_sales)   # return rows count
            .head()     # return default show 5 rows of data
                .head().T   # if too many columns and truncated
            .tail()     # last 5 rows
            .duplicated()    # return index column and boolean column of whether duplicated value
                df[~df.duplicated()]  # rows not duplicated,  keep='last' check from bottom to top
                # subset=['Harry','Ronald']  # check duplicate value in subset columns
            df = df.add_prefix('new_')   # add prefix for each column name   # add_surfix()
            .drop_duplicates()    # remove duplicated rows
            .sort_values(by=["saledate"], inplace=True, ascending=True)
            df = df.astype(dtype=np.int16)  or df['Harry'] = df['Harry'].astype(int)  # change datatype
                pd.to_numeric(df['Harry'])  # change column to numeric

            df.cumsum()  # default axis=0, from top to bottom, each cell add upper values
                        # default skipna=True  # don't consider NaN,   set to False will cause cell below NaN become NaN

            df.set_index(keys='Date', inplace=True)   # set Date column as index
            df = df.reset_index(drop=True)  # reset the index from 0 step 1, change old index to a 'index' column

            pd.concat()    # concatenate series or dictionary
                pd.concat((df1, df2), ignore_index=True)  # default axis=0, add at bottom, drop original index and
                    # assign new index start 0,1,2... (if df have default index, will cause duplicate index )
                pd.concat((df1, df2),keys=['df1','df2'], axis=0)  # add hierarchical index at level 0 'df1','df2'
                    # fill NaN and outer join in default if 2 df have different column
                pd.concat((df1, df2),join='inner', sort=True)  # inner join
                pd.concat((df1, df2),axis=0, join_axes=[df1.columns])  # left join, keep all df1 columns
                        # axis=1, join_axes=[df1.index]  # left join, keep all df1 columns

            pd.merge(df1, df2)    # merge must have common column(s), default inner join on the common column
                df1.merge(df2)   # how='inner' #inner join     how='right' # right join    outer left
                # if have multiple same values in df1 and df2, will return Cartesian product (all possiblilities )
                pd.merge(df1,df2, on='name')    # when have multiple common columns, use 'on=' to specify the column
                    # used to join, the other common column name will have extra _x, _y to distinguish
                        # suffixes=['_df1','_df2'] use surffixes to change defult _x, _y
                pd.merge(df1,df2, left_on='name', right_on='student')  # merge df1 'name' column, df2 'student' column
                pd.merge(df1,df2, left_on='name', right_index=True) # merge df1 'name' column, df2 index column

            df.drop(columns='Hermione')  # index=''  drop row,   columns=''  drop column,   inplace=False defaut,
                # labels='' + axis=0/1  drop row/column

            df.take([2,1,0])  # get the rows in order of the input list based on row index
                np.random.permutation([0,1,2,3])  # return a random permutation of array with all elements
                df.take(np.random.randint(0,len(df), size=10))   # random sampling with replacement
                df.take(np.random.permutation(np.arange(len(df)))[:10])  # random sampling without replacement

            df = df.sample(frac=0.6)    # shuffle and use 60% data, row index same, row change

            df.query('Harry=="98" & Ronald<90')   # string use "", Harry column value is "98" and Ronald column value<90
            df["Harry"]  >= 60   # return one column of index and one column of boolean of compare result
            df[df["Harry"] >= 60]   # df[index+boolean column] return rows with boolean=True

            df.rename({'Harry':'Harry Potter','Ronald':'Ron'}, axis=1)  # change column index
                # df.rename(columns={'Harry':'Harry Potter','Ronald':'Ron'})   # level=None default
                # df.rename(index={'Magic Defence':'Defence'})  # rename row index

            df.replace({34:98, 37:96, np.nan:0})  # replace 34 to 98, 37 to 96 in df, nan to 0

            df['Jennie'] = df['Harry'].map({34:98, 37:96, np.nan:0})  # create new column Jennie using column Harry with
                # mapping 34 in Harry map to Jennie 98,...
            df['Harry'] = df['Harry'].map({34:98, 37:96, np.nan:0}) # override Harry column with mapping
            df['Jennie'] = df['Harry'].map(lambda item: item*1.1)   # create column with lambda function
                # def convert(item): return item*1.2;
                # df['Jennie'] = df['Harry'].map(convert)  # pass in function for map for create Jennie column
                    # df['Jennie'] = df['Harry'].transform(convert)  # same as map passing function

            df["Date"] = pd.to_datetime(df["Date"])  # convert Date column from object to datetime64[ns]

            pd.crosstab(car_sales["Make"], car_sales["Doors"] )    #return table of 2 selected columns as x and y
            .groupby(by='Make')  # .groupby(by='Make').groups   # {'Honda':Index([0,3,8]),'Toyota':Index([1,2,6])}
                .groupby(["Make"]).mean()  #return mean value for each column for each make as row index
                .groupby(["Make"])[['Price']].mean()  # only show price column for each make

            .plot()  # default line chart (find data trend)
                plt.figure(figsize=(12, 9))   # change figure size
                plt.rcParams['font.sans-serif'] = ['SimHei']   # able to show chinese in legend
                plt.rcParams['axes.unicode_minus] = False   # solve negative sign display gibberish under chinese
                plt.set_xscale('log')   # change x axis to log scale
                plt.show()  # needed for pycharm to show plot
                    # df.plot(kind='bar')  # bar chart, comparison for each input data value
                df.plot(kind='hist', bins=50, density=True) # specify bins number, (percentage/bin witdth) as y axis
                    # check fo data distribution
                df.plot(kind='kde')  #kernel density estimate(install scipy), curve for density for input (distribution)
                df.plot(x='Harry',y='Ronald',kind='scatter')  # check 2 column/(1D) data relation

            pd.plotting.scatter.matrix(df, figsize=(16,16), alpha=0.6, diagonal='kde')
                # alpha default 0.5, [0,1] transparency, smaller more transparent
                # show all scatter plots for 2 columns combination
                # same column will show histogram default, diagonal='kde' (only kde or hist option)

            .hist() # histogram, show count for each bins for all data ranges
                plt.show()  # needed for pycharm to show plot

            df['Harry'].value_counts()  #return series with unique value in column as index and appearance time as value
                df.value_counts()  # return a series with all unique row as index, appearance time as value
                df.apply(pd.value_counts)  # return a Dataframe with all unique cell value as row index, keep original
                    # column, and count of appearance in data

            car_sales["Make"] = car_sales["Make"].str.lower()   #change make column to lower case,
                need reassign if any change of column


            car_sales["Price"] = car_sales["Price"].apply(lambda x: x*6.5)  # each price * 6.5, apply
                function to change column value


            # change string column to category, default nan category code is -1
            for label, value in df.items():
                if pd.api.types.is_string_dtype(value):
                    df["label"] = value.astype("category").cat.as_ordered()   # change string column to
                        category order by alphabet. change to int underneath,
                    df.state.cat.codes  # show int code of category column "state"
                pd.Categorical(df["state"])  #list of states (categorical)


            aggregate function
                df.sum()  #  for each column sum a value, return dataframe  (add numeric, concatenate string)
                df.sum(axis=1)  # each row sum a value(same column)
                df.sum(axis=1,level=0)  # each level 0 row index sum a value (same column)
                prod,  mean, std, var, argmin, argmax, median, abs, precentile, any, all, power

                df[~((df-df.mean()).abs() > 3 * df.std()).any(axis=1)]  # filter out row has greater than 3 std value
                df.groupby('Make')['Price'].apply(mean)    df.groupby('Make')['Price'].transform(mean)

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


            # read in and read out
            stock = pd.read_csv(os.path.join('..', 'resources', 'data', 'EH.csv')) # import csv file,
                # default sep=','   header=None (fill column name with 0,1)
                #pd.read_excel, read_html, read_json, read_sql
            # stock = pd.read_csv('http://xxx/stock.csv')
            # stock = pd.read_excel('../resources/data/EH.xlsx', sheet_name=2)
            # conn = sqlite3.connect('../resources/data/student.sqlite')
                # conn = pymsql.connect(host='127.0.0.1', port=3306, user='cai', password='123456', database=
                                # 'company', charset='utf-8'
            # pd.read_sql('select * from student limit 30', conn, index_col='stu_id')
            stock["Date"] = pd.to_datetime(stock["Date"])  # convert Date column from object to datetime64[ns]
                # stock = pd.read_csv(os.path.join('..','resources','data','EH.csv'), parse_dates=["Date"])
                    # do in one step with parse_dates
            stock["Month"] = stock["Date"].dt.month  # add a month column     stock.Date.dt.year  day,
                    # must be done before set Date as index
                stock = stock.set_index(keys='Date')  # set index to column Date, default 0,1,2...,
                    assign or set inplace=True
                print(stock.dtypes, stock.index)

                plt.figure(figsize=(12, 9))  # set figure size(inch) for plot, need set before plot(),
                    otherwise create empty figure

                stock['Adj Close'].plot()   # plot use index as x axis, and Adj Close as y axis
                plt.show()  # needed for pycharm to show plot

                stock.to_csv(os.path.join('..','resources','data','new-EH.csv'), index=False)
                    # export to csv file, excluding index column (0,1,2...)
                # stock.to_dict()
                # stock.to_json('../resources/data/new-EH.json')  Timestamp
                # stock.to_html('../resources/data/new-EH.html')
                # stock.to_sql('student_new', conn)  # if_exists='fail' default, 'ignore'  save in sqlite
                    # engine = create_engine('mysql+pymsql://cai:cai@localhost/company?charset=utf8')  # sqlalchemy lib
                    # stock.to_sql('student_new', engine)   #  save in mysql




    matplotlib
        %matplotlib inline     # show plot/figure in the jupyter notebook console
        import matplotlib.pyplot as plt
        img = plt.imread('./resources/images/hp2.jpg')   # return ndarray (matrix) of image.  need pillow if not png file
            # png [0,1] h*w*4    jpg: [0,255] h*w*3
        plt.imshow(img)  # cmap='gray' for gray scale picture
        plt.imshow(img[:,::-1])  # left right reverse
        plt.imshow(img[::2,::2])  # compress 4 pixcel become 1
        plt.show()  # add in pycharm to show all the plots declared above in one figure




        x=[1,2,3]  y=[4,5,6]   plt.plot(x,y)   # default line chart

        plt.scatter(df.age[df.target==0], df.heartrate[df.target==0], c="blue");

        #1st method
            fig = plt.figure()
            ax = fig.add_subplot()
            plt.show()

        #2nd method
            fig = plt.figure()
            ax = fig.add_axes([1,1,1,1])
            ax.plot(x,y)
            plt.show()

        # 3rd method (recommended, OOP for advanced plotting)
            fig, ax =plt.subplots(figsize=(15, 8))
            ax.plt(x, y);
            ax.set(title="Simple Plot", xlabel="x-axis", ylabel="y-axis")
            fig.savefig("images/sample_plot.png")

        subplots:
            #option 1
                fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,5))
                ax1.plot(x,y)
            #option 2
                fig, ax =plt.subplots(nrows=2, ncols=2, figsize=(10,5), sharex=True)
                    #share x label remove xlabel
                ax[0, 0].plot(x,y)

        ax.scatter(x, y)    # scatter plot
        ax.bar(dict.keys(), dict.values())
        ax.hbar(list(dict.keys()), list(dict.values()))
        ax.hist(x)

        df = pd.Series(np.random.randn(1000), index=pd.date_range)"1/1/2020", periods=1000))
        df=t.cumsum()
        df.plot(x="column name 1",y="column name 1")
        df.plot.bar()
        df.plot.hist(bins=10, figsize=(10,30),subplots=True)  #plot all columns, same scale
        plt.legend(["Female", "Male"])
        plt.xticks(rotation=0)  #rotate the scale ticks on x-axis
        plt.xticks(np.arrange(0, 101, 10))

        #correlation matrix heat map
        corr_mat = df.corr()  #correlation matrix for all columns vs all columns, max value 1, min value -1,
            # positive, negative correlation, higher absolute value more correlation
        fig, ax = plt.subplots(figsize=(15,10))
        ax = sns.heatmap(corr_mat, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")

        #mix pandas and oop
        fig, ax =plt.subplots(figsize=(15, 8))
        ax.plt(kind='scatter', x="age", y="chol",c="target", ax=ax);
        ax.set_xlim([45,100])
        fig.savefig("images/sample_plot.png")

        #oop
        fig, ax =plt.subplots(figsize=(15, 8))
        scatter = ax.scatter(x=df["age"], y=df["chol"],c=df["target"], cmap="winter");  # cmap change color scheme
        ax.set(title="Heart disease", xlabel="Age", ylabel="Cholesterol")
        ax.legend(*scatter.legend_elements(), title="Target")
        ax.axhline(y, linestyle="--", color="blue")
        fig.suptitle("Overall title", fontsize=16, fontweight="bold")
        ax.legend().set_visible(True)

        plt.style.available   # show available plot style
        plt.style.use('seaborn-whitegrid')

        #方法1：先创建窗口，再创建子图。（一定绘制）
        fig = plt.figure(num=1, figsize=(15, 8),dpi=100)     #开启一个窗口，同时设置大小，分辨率
        ax1 = fig.add_subplot(2,1,1)  #通过fig添加子图，参数：行数，列数，第几个。
        ax2 = fig.add_subplot(2,1,2)  #通过fig添加子图，参数：行数，列数，第几个。
        print(fig,ax1,ax2)
        #方法2：一次性创建窗口和多个子图。（空白不绘制）
        fig,axarr = plt.subplots(4,1)  #开一个新窗口，并添加4个子图，返回子图数组
        ax1 = axarr[0]    #通过子图数组获取一个子图
        print(fig,ax1)
        #方法3：一次性创建窗口和一个子图。（空白不绘制）
        ax1 = plt.subplot(1,1,1,facecolor='white')      #开一个新窗口，创建1个子图。facecolor设置背景颜色
        print(ax1)
        #获取对窗口的引用，适用于上面三种方法
        # fig = plt.gcf()   #获得当前figure
        # fig=ax1.figure   #获得指定子图所属窗口


    Scipy
        Fourier transform and assign value for points greater than threshold, then inverse Fourier transform and take
            real part as final result, thresh hold greater remove more noise point and image will have less content
        from scipy.fftpack import fft2, iff2   # Fourier transform;
        moon = Image.open('../resources/images/moon.png')
        moon = moon.filter(ImageFilter.MedianFilter)  # CONTOUR, BLUR, GaussianBlur, SMOOTH SHARPEN, MaxFilter, EMBOSS,
            #  EDGE_ENHANCE, DETAIL, FIND_EDGES
        moon.convert('L').save('../resources/images/moon_gray.png')  # convert to gray scale image
        moon = plt.imread('../resources/images/moon.jpg')
        moon_fft = fft2(moon)  # Fourier transform   [[-1.35476888e+02 - 7.19055378e+02j...   shape (824, 1203)
                        # absolute value greater than threshold are noise point
        # moon_fft[np.abs(moon_fft) > 8e2] = 0;   result = moon_fft
        result = np.where(np.abs(moon_fft) > 1e3, 0, moon_fft)  # assign 0 in Fourier transform with large value
        moon_ifft = ifft2(result)  # inverse Fourier transform
        moon_cleaned = np.real(moon_ifft)  # only keep real part of array
        plt.imshow(moon_cleaned, cmap='gray')
        plt.show()


        # integration
        half_pi, deviation = integrate.quad(lambda x:(1-x**2)**0.5, -1, 1)    # inteegration of y= (1-x^2)^0.5  xϵ[-1,1]
            # get pi from r=1 (0,0) circle
        print(half_pi*2, deviation)   # deviation: error range


        # read, write data into .mat file
        io.savemat('../resources/data/moon.mat', {'moon':moon_cleaned})  # save in byte .mat file, specify dict key: value
        data = io.loadmat('../resources/data/moon.mat')  # return dict
        plt.imshow(data['moon'], cmap='gray')
        plt.show()

        # image operation
        face = misc.face()  # gray=False default,  True return gray scale image
        face_shift = ndimage.shift(face, (100,200,0), mode='constant')     # x, y, color   # move image left 100, down 200
            # mode='reflect'/'mirror' (reflect from image from boundary); default constant (0 (black) for empty space after
                # move);  'nearest' (expand last pixel value at boundary to fill rest empty area); 'wrap' (move the out of
                # boundary image part to the other side of empty space)
        face_rotate = ndimage.rotate(face_shift, angle=30)  # counterclockwise 30 degree
        face_zoom = ndimage.zoom(face_rotate, (0.5, 0.5, 1)) # zoom image width 0.5 times, height 0.5 times, color same
        face_slice = face_zoom[180:320, 400:600]  # slice from top 180 px to 320 px. from left 400 to 600px
        plt.imshow(face_slice)
            # add cmap='gray' if gray scale image
'''
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from pandas import Timestamp
from scipy import integrate, io, misc, ndimage, signal
from scipy.fftpack import fft2, ifft2


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
    print(df.sum().iloc[0:2])
    print(df[df['Male','Harry'] >= 60])


    stock = pd.read_csv(os.path.join('..', 'resources', 'data', 'EH.csv')) # import csv file,
        # default sep=','   header=None (fill column name with 0,1)
        #pd.read_excel, read_html, read_json, read_sql
    # stock = pd.read_csv('http://xxx/stock.csv')
    # stock = pd.read_excel('../resources/data/EH.xlsx', sheet_name=2)
    # conn = sqlite3.connect('../resources/data/student.sqlite')
        # conn = pymsql.connect(host='127.0.0.1', port=3306, user='cai', password='123456', database='company',
                    # charset='utf-8'
    # pd.read_sql('select * from student limit 30', conn, index_col='stu_id')
    stock["Date"] = pd.to_datetime(stock["Date"])  # convert Date column from object to datetime64[ns]
        # stock = pd.read_csv(os.path.join('..','resources','data','EH.csv'), parse_dates=["Date"])
            # do in one step with parse_dates
    stock["Month"] = stock["Date"].dt.month  # add a month column     stock.Date.dt.year  day,
        # must be done before set Date as index
    stock = stock.set_index(keys='Date')  # set index to column Date, default 0,1,2..., assign or set inplace=True
    print(stock.dtypes, stock.index)

    plt.figure(figsize=(12, 9))  # set figure size(inch) for plot, need set before plot(), otherwise create empty figure
    stock['Adj Close'].plot()   # plot use index as x axis, and Adj Close as y axis
    plt.show()  # needed for pycharm to show plot

    stock.to_csv(os.path.join('..','resources','data','new-EH.csv'), index=False)
        # export to csv file, excluding index column (0,1,2...)
    # stock.to_dict()
    # stock.to_json('../resources/data/new-EH.json')  Timestamp
    # stock.to_html('../resources/data/new-EH.html')
    # stock.to_sql('student_new', conn)  # if_exists='fail' default, 'ignore'  save in sqlite
        # engine = create_engine('mysql+pymsql://cai:cai@localhost/company?charset=utf8')  # sqlalchemy lib
        # stock.to_sql('student_new', engine)   #  save in mysql
    print(stock.head())

    df3 = pd.DataFrame({'price': [10, 30], 'color': ['red', 'blue']}, index=['A', 'B'])
    df3=df3.add_prefix('car_')
    print(df3)


def matplotlib_basic():
    img = plt.imread('../resources/images/hp2.jpg')  # return ndarray (matrix) of image.  need pillow if not png file
    # png [0,1] h*w*4    jpg: [0,255] h*w*3
    #plt.imshow(img)
    plt.imshow(img[:, ::-1])  # left right reverse
    plt.show()
    pd.Series(np.random.randint(0, 10, size=10)).plot()
    plt.show()

def scipy_basic():
    # Use Fourier transform
    moon = plt.imread('../resources/images/moon.png')  # shape (824, 1203, 3)
    moon = Image.open('../resources/images/moon.png')
    moon = moon.filter(ImageFilter.MedianFilter)  # CONTOUR, BLUR, GaussianBlur, SMOOTH SHARPEN, MaxFilter
    moon.convert('L').save('../resources/images/moon_gray.png')
    moon = plt.imread('../resources/images/moon_gray.png') # shape (824, 1203)
    plt.imshow(moon, cmap='gray')
    #plt.show()  # needed for pycharm to show plot

    moon_fft = fft2(moon)  # Fourier transform   [[-1.35476888e+02 - 7.19055378e+02j...   shape (824, 1203)
                    # absolute value greater than threshold are noise point
    #print(moon_fft)
    moon_fft[np.abs(moon_fft) > 1e5] = 0
    result =moon_fft
    #result = np.where(np.abs(moon_fft) > 1e3, 0, moon_fft)  # assign 0 in Fourier transform with large value
        # thresh hold greater remove more noise point and image will have less content
    moon_ifft = ifft2(result)  # inverse Fourier transform
    moon_cleaned = np.real(moon_ifft)  # only keep real part of array

    plt.imshow(moon_cleaned, cmap='gray')
    #plt.show()


    # integration
    half_pi, deviation = integrate.quad(lambda x:(1-x**2)**0.5, -1, 1)    # inteegration of y= (1-x^2)^0.5  xϵ[-1,1]
        # get pi from r=1 (0,0) circle
    print(half_pi*2, deviation)   # deviation: error range

    # read, write data into .mat file
    io.savemat('../resources/data/moon.mat', {'moon':moon_cleaned})  # save in byte .mat file, specify dict key: value
    data = io.loadmat('../resources/data/moon.mat')  # return dict
    plt.imshow(data['moon'], cmap='gray')
    #plt.show()

    # image operation
    face = misc.face()  # gray=False default,  True return gray scale image
        # face = plt.imread('../resources/images/xxx.png')
    face_shift = ndimage.shift(face, (100,200,0), mode='constant')     # x, y, color   # move image left 100, down 200
        # mode='reflect'/'mirror' (reflect from image from boundary); default constant (0 (black) for empty space after
            # move);  'nearest' (expand last pixel value at boundary to fill rest empty area); 'wrap' (move the out of
            # boundary image part to the other side of empty space)
    face_rotate = ndimage.rotate(face_shift, angle=30)  # counterclockwise 30 degree
    face_zoom = ndimage.zoom(face_rotate, (0.5, 0.5, 1)) # zoom image width 0.5 times, height 0.5 times, color same
    face_slice = face_zoom[180:320, 400:600]  # slice from top 180 px to 320 px. from left 400 to 600px
    plt.imshow(face_slice)
        # add cmap='gray' if gray scale image
    #plt.show()

    # generate noisy image (color image)
    face_noise = np.array(face)
    face_noise = face_noise.astype(np.int32)   # default unit8, cause issue when runs beyond 255 or below 0
    add = np.floor(np.random.randn(*face_noise.shape) * face_noise.std() * 0.5)
    add = add.astype(np.int32)
    face_noise = face_noise + add
    face_noise = np.where(face_noise > 255, 255, face_noise)  # trim above 255 to 255
    face_noise = np.where(face_noise < 0 , 0, face_noise)  # trim below 0 to 0
    face_noise = face_noise.astype(np.uint8)
    plt.imshow(face_noise)
    #plt.show()

    # generate noisy image (gray scale image)
    face2 = misc.face(gray=True)
    face_noise2 = face2.astype('float64')  # default unit8, cause issue when runs beyond 255 or below 0

    face_noise2 += np.random.randn(*face_noise2.shape) * face_noise2.std() * 0.5
    face_noise2 = np.where(face_noise2 > 255, 255, face_noise2)  # trim above 255 to 255
    face_noise2 = np.where(face_noise2 < 0, 0, face_noise2)  # trim below 0 to 0
    face_noise2 = face_noise2.astype(np.uint8)
    plt.imshow(face_noise2, cmap='gray')
    #plt.show()



    # gaussian  sigma bigger cause blur (lose detail), small cause miss noise
    face_gaussian = ndimage.gaussian_filter(face_noise,sigma=1)
    plt.imshow(face_gaussian)
    #plt.show()

    # median filter
    face_median = ndimage.median_filter(face_noise, size=4)
    plt.imshow(face_median)
    #plt.show()

    # wiener filter  (not good)
    face_wiener = signal.wiener(face_noise2, mysize=5)
    plt.imshow(face_wiener, cmap='gray')
    #plt.show()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    ax1.imshow(face)
    ax2.imshow(face_noise)
    ax3.imshow(face_gaussian)
    ax4.imshow(face_median)
    plt.show()

if __name__ == '__main__':
    # numpy_basic()
    # pandas_basic()
    #matplotlib_basic()
    scipy_basic()
