'''
    pip install sklearn

    KNN (k nearest neighbor)
        find k nearest nodes to the test point, assign majority class of those k nodes

        KNeighborsClassifier()   # can handle string target column, transfer target column to categorical optional
            # n_neighbors=5
            # weights='uniform'  # weights for different point
            # leaf_size=30
            # p=2   # l2 (euclidean_distance)     p=1:  l1 (manhattan_distance)
            n_jobs=None   #number of parallel jobs (process) to run for neighbors search

                count = 3  # test data count
                knn = KNeighborsClassifier(n_neighbors=3)  # default 5 neighbor, small dataset here
                X_train = df.iloc[:-count, 1:-1]   # first column title not used for train, last column type is label
                y_train = df.iloc[:-count, -1]  # last column target column
                X_test = df.iloc[-count:, 1:-1]  # last count rows for testing
                y_test = df.iloc[-count:, -1]
                data, target = df.iloc[:, 1:-1], df.iloc[:, -1]
                X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2) # shuffle=True default
                    # built in function to split into train, test
                knn.fit(X_train, y_train)  # train model

                print(knn.predict(X_test))   # 2D array (dataframe/[[]]) as input
                print(knn.score(X_train, y_train), knn.score(X_test, y_test))



'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn_basic():
    df = pd.read_excel('../resources/data/movie.xlsx', sheet_name='movie')

    # change string column to category, default nan category code is -1
    df["type"] = df["type"].astype("category") # change string column to category
        #df["type"] = df["type"].astype("category").cat.as_ordered()  # category order by alphabet
    print(pd.Categorical(df["type"]))  # list of states (categorical)

    #plt.scatter(df['kiss'], df['fight'], c=[0,0,1,0,1,1,0,0])
    plt.scatter(df['kiss'], df['fight'], c=df.type.cat.codes.tolist())
        # df.type.cat.codes    return series int code of category column "type"
    #plt.show()


    count = 3  # test data count
    knn = KNeighborsClassifier(n_neighbors=3)  # default 5 neighbor, small dataset here
    X_train = df.iloc[:-count, 1:-1]   # first column title not used for train, last column type is label
    y_train = df.iloc[:-count, -1]  # last column target column
    X_test = df.iloc[-count:, 1:-1]  # last count rows for testing
    y_test = df.iloc[-count:, -1]
    data, target = df.iloc[:, 1:-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2) # shuffle=True default
        # built in function to split into train, test
    knn.fit(X_train, y_train)  # train model

    print(knn.predict(X_test))   # 2D array (dataframe/[[]]) as input
    print(knn.score(X_train, y_train), knn.score(X_test, y_test))

if __name__ == '__main__':
    knn_basic()