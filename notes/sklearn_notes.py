'''
    Scikit-learn
        mainly for machine learning, not for deep learning, don't support GPU
    pip install sklearn

    KNN (k nearest neighbor)
        find k nearest nodes to the test point, assign majority class of those k nodes

        KNeighborsClassifier()   # can handle string target column, not string attribute column
            # n_neighbors=5
            # weights='uniform'  # weights for different point
            # leaf_size=30
            # p=2   # l2 (euclidean_distance)     p=1:  l1 (manhattan_distance)
            n_jobs=None   #number of parallel jobs (process) to run for neighbors search

        tree = DecisionTreeClassifier()
            # Decision tree easily cause overfitting, use trim branch to avoid
            # pre trim
            # max_depth=4,  # default None till pure leaf, max depth control overfitting. underfitting
            # min_samples_split=4    # default 2, if less than 2 items in node, stop split
                if decimal, consider as percentage
            # min_samples_leaf=2      # default 1, if 1 items in node, stop split
        tree.fit(X_train, y_train)

        random forest
            random samples(same total) for each tree, random feature column for different tree
        rfc=RandomForestClassifier()
            # n_estimators  # number of trees in forest default 100
            rfc.fit(X_train, y_train)
            rfc.feature_importances_

        LogisticRegression()  # bad for too many features
            # prediction very stable, generally good result
            lr = LogisticRegression(max_iter=1000)
                # C  # penalty coefficient default 1.0, positive float, smaller stronger regularization
                # solver='lbgfs' default
                # liblinear  for small data set   L1/L2 loss
                # lbgfs, sag, newton-cg for large multiclass dataset    L2 loss
                # saga for huge dataset multiclass dataset   Elastic-Net regularization
            lr.fit(X_train, y_train)


        LinearSVC()   svm classifier
            better performance than SVC(kernel='linear')
            # prediction not stable
            SVM: small data set regression/ classification
            svc = LinearSVC()   # can handle string target column, not data string column
            # rbf
            # gamma  coefficient for nonlinear kernel. greater gamma, more complicate model and more overfitting
            # C  # penalty coefficient default 1.0, positive float, smaller stronger regularization
            # kernel = 'rbf' (radical based kernel function,decision boundary circular arc) default, 'linear',
                'poly'(decision boundary polynomial) 'sigmoid'
            svc.get_params()  # get input param
            svc.set_params(key=value)  # set input param
            svc.fit(X_train, y_train)  # train model
            print(svc.predict(X_test))  # 2D array (dataframe/[[]]) as input
            print(svc.score(X_train, y_train), svc.score(X_test, y_test))
            svc.support_vectors_
            svc.coef_,  svc.intercept_
            svc.decision_function((x,y))  # return distance of point to decision hyperplane

        LinearSVR()  svm regressor, regression score closer to 1 better
            svr = LinearSVR()
            svr.fit(X_train, y_train)  # train model
            print(svr.predict(X_test))  # 2D array (dataframe/[[]]) as input
            print(svr.score(X_train, y_train), svr.score(X_test, y_test))

        LinearRegression()  linear regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)  # train model
            print(lr.predict(X_test))  # 2D array (dataframe/[[]]) as input
            print(lr.score(X_train, y_train), lr.score(X_test, y_test))

        Ridge()     linear regression with L2 regularization
                    # ridge regressor, eliminate some coefficient which are too sensitive or not sensitive
                    # same as linear regression, but add second order regularization, able to inverse the originally
                    # singular matrix, help reduce overfitting (bigger λ, smaller θ)
                    # θ= (X^T∙X+λI)^(−1)∙X^T∙y
            ridge = Ridge(alpha=0.1)    # alpha=1 default is λ in the equation   [0.1-0.00001]
            ridge.fit(X,y)

        Lasso()     linear regression with L1 regularization
            # least absolute shrinkage and selection operator
            # total of coeffecient sum less than λ   first order regularization
            lasso = Lasso(alpha=0.1)
            lasso.fit(X,y)

        KNeighborsRegressor()  K nearest neighbor regressor
            ridge = Ridge(alpha=0.1)    # alpha=1 default is λ in the equation   [0.1-0.00001]
            ridge.fit(X,y)

        KMeans()
            # ISODATA algorithm get desired k
            # if data is dirty, cause long slim shape clustering, or each column has different std, or each cluster have
                # huge difference in counts, those can cause less accurate result

            # n_clusters=3  # default 8 centers to cluster
            km = KMeans(n_clusters=3)
            km.fit(data)
            km.cluster_centers    # np array of centers coordinates
            km.labels_   # array of label of center belong to for data


        Bayes: only for classification
            GaussianNB(): continuous data, data gaussian distribution
            MultinomialNB(): positive discrete data, multinomial distribution
            BernoulliNB(): binary discrete data, bernoulli distribution

        Categorical data transform
        # map approach
        df['Sex'] = df['Sex'].map({'female':0, 'male':1})

        # category codes approach
        for s in categorical_features:
            data[s] = data[s].astype('category').cat.as_ordered()   # converting type of columns to 'category', nan code is -1
            data[s] = data[s].cat.codes   # Assigning numerical values to the column
                # df[s+'_cat'] = df[s].cat.codes  # or assign to new column

        # label encoder approach
        labelencoder = LabelEncoder()  # creating instance of labelencoder
        for s in categorical_features:
            data[s] = labelencoder.fit_transform(data[s])   # Assigning string column with numerical values
            # data[s+'_org'] = labelencoder.inverse_transform(data[s])  # inverse_transform convert int to string

        # pd.get_dummies approach
        data = pd.get_dummies(data[categorical_features])

        # OneHotEncoder + ColumnTransformer approach
        transformer = ColumnTransformer([("one_hot", OneHotEncoder(), categorical_features)], remainder="passthrough")
        data = np.array(transformer.fit_transform(data))  #, dtype=np.str
        #data = pd.DataFrame(data)
        #print(data.head())



        Normalization (min-max scaling)  x=(x-x.min)/(x.max-x.min)
            This rescales all the numerical values to between 0 and 1, set feature_range=[0,2] to change range
            not recommended for large dataset,(affect greatly by outlier)
            scaler = MinMaxScaler(feature_range=[0,1], copy=False)  #copy false do inplace change
        Standardization
            This subtracts the mean value from all of the features (have 0 mean), divide by std
            scaler = StandardScaler()
            scaler.fit(data); scaler.transform(data)
            # or use fit_transform(data)   # transformed data std=1




        save model using pickle or joblib
            # using pickle
            pickle.dump(clf, open("random_forest.pkl", "wb"))
            loaded_model = pickle.load("random_forest.pkl", "rb"))
            loaded_model.score(X_test, y_test)

            #using joblib, less compatible, faster with numpy estimator
            from sklearn.exernals import joblib
            joblib.dump(clf, filename="random_forest.joblib")
            loaded_model = joblib.load("random_forest.joblib")
            loaded_model.score(X_test, y_test)




        GridSearchCV and RandomizedSearchCV
        grid = {"n_estimators" : [10, 100, 200, 500, 1000, 1200], "max_depth" : [None, 5, 10, 20, 30],
        "max_feature" : ["auto", "sqrt"], "min_samples_split" : [2, 4, 6], "min_samples_leaf" :[1, 2, 4]}
        heart_shuf = heart_disease.sample(frac=1)
        rs_clf = RandomizedSearchCV(estimator=clf, param_grid=grid, n_iter=10, cv=5,
            verbose=2 )    #  n_iter: number of models to try
        rs_clf.fit(X_train, y_train)
        rs_clf.best_params_
        y_pred = rs_clf.predict(X_test)
        accuracy_score(y_test, y_pred)

        GridSearchCV (brute force all combination)
            smaller the grid using reference from randomsearchcv
        gs_clf = GridSearchCV(estimator=clf, param_distributions=grid, cv=5, verbose=2 )
            # cv: cross validation (split training data to 80% train and 20% validation rotation)
            gs_clf.best_params_


        pca = PCA(n_components=32, whiten=True)
            # n_components: reduce data to n dimension,  whiten: make std in each column same
            after transformation, lose physical meaning
            X_train_pca = pca.fit_transform(X_train)


        Evaluation classification
            True positive = model predict 1 when truth is 1
            False positive = model predict 1 when truth is 0
            ROC use a comparison of a model's true  positive rate (x axis) versus false positive rate(y axis)
            precision = true pos /  (true pos+ false pos)
            recall = true pos / (true pos + false neg)    # more important if imbalanced data
            f1 score = 2*precision*recall / (precision+recall) = TP / (TP + 0.5*(FP+FN))
            support: number of samples each metrics is calculated on
            accuracy: accuracy of model (# correct predictions / total predictions)
        Evaluation regression
            R^2: compare prediction to the mean of the target
                R^2 = 1-RSS/TSS  [−∞,1]
                RSS(sum of residuals)  RSS=∑_i(y_i−f(x_i ))^2   sum of (target-predict)^2
                TSS(total sum of squares)   RSS=∑_i(y_i−avg(y))^2   sum of (target-target mean)^2
            Mean absolute error (MAE): average absolute difference between predicted and actual
                value, show how wrong prediction
            Mean squared error (MSE): average of squared difference between predicted and
                actual value, amplifies larger difference

            cross_val_score(clf, X, y, cv=5)  # score by 5 fold cross-validation
                # classifier, data, target, cross validation fold
                kf = KFold(5)  # k fold(test on first 1/5, train on next 4/5, shifting down)
                kf = StratifiedKFold(5)   # k fold(test on first 1st and 6th 1/10, , train on rest shifting down)
                for train, test in kf.split(X,y)   # return array of train and test index for k iteration
            accuracy_score(y_test, y_pred)
            precision_score(y_test, y_pred)
            recall_score(y_test, y_pred)
            f1_score(y_test, y_pred)
            conf_mat = confusion_matrix(y_test, y_preds)  #  [true neg, false pos] [false neg, true pos]


            prob=clf.predict_proba(X)  # return probability of each class
                # SVC(probability=True)  # add probability=True to enable predict_proba
            fpr, tpr, threshold = roc_curve(y_test, prob[:,1])
                #    probability of positive
            plt.plot(fpr,tpr)  # roc curve
            auc(fpr_mean, tpr_mean)

            r2_score(y_test, y_pred)
            mean_absolute_error(y_test, y_pred)  # avg difference
            mean_squared_error(y_test, y_pred)  # avg squared difference


        Linear difference
            x and y are arrays of values used to approximate some function f: y = f(x).
            This class returns a function whose call method uses interpolation to find the value of new points.
            # using know to generate data based on trained data distribution pattern
            from scipy.interpolate import interp1d
            f = interp1d(x, y)
            new_y = f(new_x) # generate new_y, given new_x, based on old x,y relationship


        check column value distribution for different class, if different distribution, then it's a useful feature
            df = pd.DataFrame(data)
            c1_0 = df.iloc[target==0, 0]  # get first column values with class=0
            c1_1 = df.iloc[target == 1, 0]
            c1_0.plot(kind='hist', density=True, bins=50, alpha=0.5)  # blue hist
            c1_1.plot(kind='hist', density=True, bins=100, alpha=0.5)  # orange hist
            plt.show()

        imbalanced class
            get more data for minority class; modify loss function weights to penalize the incorrect prediction of
            minority class; undersampling of majority class; oversampling of minority class using smote (synthetic
            minority oversampling technique)(generate data by: get total k centers for minor class data, randomly
            pick n centers, use linear interpolation to generate new data and combined with old minor class data )

            pip install imbalanced_learn
            smote = SMOTE()
            data_resampled, target_resampled = smote.fit_resample(data,target)  # equal count for each class

'''
import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes, load_iris, load_sample_image
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import confusion_matrix, silhouette_score, roc_curve, auc, f1_score, recall_score, precision_score, \
    accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, SVR, SVC
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.interpolate import interp1d


def classifier_basic():
    #df = pd.read_excel('../resources/data/movie.xlsx', sheet_name='movie')
    df = pd.read_csv('../resources/data/Iris.csv')

    # change string column to category, default nan category code is -1
    df[df.columns[-1]] = df[df.columns[-1]].astype("category") # change string column to category
        #df["type"] = df["type"].astype("category").cat.as_ordered()  # category order by alphabet
    df[df.columns[-1]] = df[df.columns[-1]].cat.codes  # assign int to category column
    print(pd.Categorical(df[df.columns[-1]]))  # list of states (categorical)

    #plt.scatter(df['kiss'], df['fight'], c=[0,0,1,0,1,1,0,0])
    #plt.scatter(df['kiss'], df['fight'], c=df.type.cat.codes.tolist())
        # df.type.cat.codes    return series int code of category column "type"
    #plt.show()

    df.iloc[:, 1:].plot()
    plt.show()



    count = 3  # test data count
    
    X_train = df.iloc[:-count, 1:-1]   # first column title not used for train, last column type is label
    y_train = df.iloc[:-count, -1]  # last column target column
    X_test = df.iloc[-count:, 1:-1]  # last count rows for testing
    y_test = df.iloc[-count:, -1]
    data, target = df.iloc[:, 1:-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2) # shuffle=True default
        # built in function to split into train, test.   dframe and series for x and y
        # test_size =0.2  20% test,   if integer test_size = 50 then 50 test data
        # random_state=1      use together with np.random.seed(8)

    knn = KNeighborsClassifier(n_neighbors=5)  # default 5 neighbor, small dataset here
    knn.fit(X_train, y_train)  # train model
    y_ = knn.predict(X_test)  # 2D array (dataframe/[[]]) as input, return ndarray
    print(knn.score(X_train, y_train), knn.score(X_test, y_test))
    print(confusion_matrix(y_test, y_))

    svc = SVC(kernel='linear')
    svc.fit(X_train, y_train)  # train model
    print(svc.predict(X_test))  # 2D array (dataframe/[[]]) as input
    print(svc.score(X_train, y_train), svc.score(X_test, y_test))


    # visualization: show test result with line chart, every column as x axis
    plt.figure(figsize=(4*4, 3*6))  # 4*3 plots   4*6 size
    print(type(X_test),type(y_),type(y_test),y_.shape )
    for i in range(14):
        axes = plt.subplot(6,4,i+1)
        axes.plot(X_test.iloc[i*2], marker='o', ls='None')
        if y_test.iloc[i*2] != y_[i*2]:
            axes.set_title('True: %s\nPredict: %s' % (y_test.iloc[i*2],y_[i*2]), fontdict=dict(fontsize=20, color='r'))
        axes.set_title('True: %s\nPredict: %s' % (y_test.iloc[i*2], y_[i * 2]))
    plt.show()



    # visualize with using 2 columns, show decision boundary
    x = np.linspace(df.iloc[:, 1].min(), df.iloc[:, 1].max(), 1000)
    y =np.linspace(df.iloc[:, 2].min(), df.iloc[:, 2].max(), 1000)
    X, Y = np.meshgrid(x, y)
    XY = np.c_[X.ravel(), Y.ravel()]  # X.ravel()  return 1d array,   np.c_ join together X,Y with result rows as both
    # XY get all coordinates of mesh crossing points

    knn.fit(X_train.iloc[:, 0:2], y_train)   # only use first 2 column to train for easier visualization
    y_=knn.predict(XY)

    #plt.scatter(XY[:, 0], XY[:, 1], c=y_)  # slower
    plt.pcolormesh(X, Y, y_.reshape(1000,1000))  # faster
    plt.scatter(df.iloc[:, 1], df.iloc[:, 2], c=pd.Categorical(df[df.columns[-1]]), cmap='rainbow')
    plt.show()





def regressor_basic():
    np.random.seed(8)
    #df = pd.read_excel('../resources/data/movie.xlsx', sheet_name='movie')
    df = pd.read_csv('../resources/data/insurance.csv')
    df.dropna(subset=[df.columns[-1]], inplace=True)

    categorical_features = ['sex','smoker','region']
    data, target = df.iloc[:, 1:-1], df.iloc[:, -1]

    # category codes approach
    for s in categorical_features:
        data[s] = data[s].astype('category').cat.as_ordered()   # converting type of columns to 'category', nan code is -1
        data[s] = data[s].cat.codes   # Assigning numerical values to the column
            # df[s+'_cat'] = df[s].cat.codes  # or assign to new column
    '''
    # label encoder approach
    labelencoder = LabelEncoder()  # creating instance of labelencoder
    for s in categorical_features:
        data[s] = labelencoder.fit_transform(data[s])   # Assigning string column with numerical values
        # data[s+'_org'] = labelencoder.inverse_transform(data[s])  # inverse_transform convert int to string
    
    # pd.get_dummies approach
    data = pd.get_dummies(data[categorical_features])
     
    # OneHotEncoder + ColumnTransformer approach
    transformer = ColumnTransformer([("one_hot", OneHotEncoder(), categorical_features)], remainder="passthrough")
    data = np.array(transformer.fit_transform(data))#, dtype=np.str
    #data = pd.DataFrame(data)
    #print(data.head())
    '''

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2) # shuffle=True default
        # built in function to split into train, test. data, target can be df or series or np.array
    # pd.plotting.scatter_matrix(data, figsize=(16, 16), alpha=0.6, diagonal='kde')
    # plt.show()
    '''
    estimators = {'svr':SVR(kernel='rbf'),'lasso':Lasso(alpha=0.01)}
    for k, est in estimators.items():
        est.fit(X_train, y_train)
        print(est.score(X_train, y_train), est.score(X_test, y_test))
    '''
    svr = SVR(kernel='rbf')  # kernel='rbf'(radial basis function) 'linear' 'poly' 'sigmoid'
    svr.fit(X_train, y_train)  # train model
    print(svr.predict(X_test))  # 2D array (dataframe/[[]]) as input
    print(svr.score(X_train, y_train), svr.score(X_test, y_test))

    lr = LinearRegression()
    lr.fit(X_train, y_train)  # train model
    print(lr.predict(X_test))  # 2D array (dataframe/[[]]) as input
    print(lr.score(X_train, y_train), lr.score(X_test, y_test))


    x = np.random.rand(100)*10   # 100 points range [0,10]
    y = np.sin(x)
    y[::4] += np.random.randn(25)*0.2  # add noise every 4 points
    knn = KNeighborsRegressor()
    knn.fit(x.reshape(-1, 1), y)
    X_test = np.linspace(0,15,151).reshape(-1, 1)  # reshape to 101*1
    y_ = knn.predict(X_test)
    plt.scatter(x, y)
    plt.plot(X_test, y_, c='r')
    plt.show()


def cifar10():
    #a=plt.imread('../resources/data/cifar10/train/airplane/0009.png')
    #plt.imshow(a)
    #plt.show()


    X_train, X_test, y_train, y_test = [],[],[],[]

    for category in os.listdir('../resources/data/cifar10/train'):
        for name in os.listdir(os.path.join('../resources/data/cifar10/train',category)):
            temp = plt.imread(os.path.join('../resources/data/cifar10/train/', category, name))
            X_train.append(temp)
            y_train.append(category)

    for category in os.listdir('../resources/data/cifar10/test'):
        for name in os.listdir(os.path.join('../resources/data/cifar10/test',category)):
            temp = plt.imread(os.path.join('../resources/data/cifar10/test/', category, name))
            X_test.append(temp)
            y_test.append(category)

    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    X_train, X_test = X_train.reshape(X_train.shape[0], -1),  X_test.reshape(X_test.shape[0], -1)

    #knn = KNeighborsClassifier()

    pca = PCA(n_components=32, whiten=True)  # n_components: dimension,  whiten: make std in each column same
    X_train_pca = pca.fit_transform(X_train)
    print(X_train_pca.shape)

    if os.path.exists('../resources/cifar10.pkl'):
        svc = pickle.load("random_forest.pkl", "rb")
    else:
        svc = SVC(kernel='rbf')
        svc.fit(X_train_pca, y_train)
        pickle.dump(svc, open("../resources/random_forest.pkl", "wb"))


    y_pred = svc.predict(pca.transform(X_test))
    print(svc.score(pca.transform(X_train),y_train), svc.score(pca.transform(X_test), y_test))
    '''
    #lr = LogisticRegression(solver='newton-cg',max_iter=10) #solver='newton-cg',max_iter=100
    #lr.fit(X_train, y_train)
    #y_pred = lr.predict(X_test)
    #print(lr.score(X_train, y_train), lr.score(X_test, y_test))
    '''

    plt.figure(figsize=(10*2, 10*3))
    for i in range(99):
        axes = plt.subplot(10,10,i+1)
        axes.imshow(X_test[i*100].reshape(32,32,3))
        axes.axis('off')
        if y_test[i*100] != y_pred[i*100]:
            axes.set_title('True: %s\nPredict: %s' % (y_test[i*100], y_pred[i*100]),
                           fontdict=dict(fontsize=12, color='r'))
            axes.set_title('True: %s\nPredict: %s' % (y_test[i*100], y_pred[i*100]))
    plt.show()


def linear_regression():
    np.random.seed(8)
    theta_true = np.array([3,4])
    data = np.random.rand(1000, 2) * 10   # range [0,10]  1000 row data 2 column
    target = np.dot(data, theta_true.T) + 5
    target[::4] += np.random.randn(250)*0.2
    lamda = 0.001
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)
    lr = LinearRegression()
    lr.fit(X_train, y_train)  # train model
    #print(lr.predict(X_test))  # 2D array (dataframe/[[]]) as input
    print(lr.score(X_train, y_train), lr.score(X_test, y_test))
    print('0',lr.coef_, lr.intercept_)  # [2.99836213 3.9982835 ] 5.017313628480423



    # print(lr.predict(X_test))  # 2D array (dataframe/[[]]) as input
    print(lr.score(X_train, y_train), lr.score(X_test, y_test))
    print('0', lr.coef_, lr.intercept_)  # [2.99836213 3.9982835 ] 5.017313628480423

    # mathematical analytical solution
    #lamda = 0.0001  # regularization term
    X = pd.DataFrame(X_train)
    X['const'] = np.full(X.shape[0],1.0)  # add coefficient column all 1
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + lamda * np.identity(X.T.shape[0])),X.T),y_train)
        #θ= (X^T∙X)^(−1)∙X^T∙y  no regularization     # θ= (X^T∙X+λI)^(−1)∙X^T∙y  Ridge regression
    print('1',theta)   # [2.99836525 3.99828673 5.01727547]

    # stochastic gradient decent
    learning_rate = 0.01
    theta2 = np.random.randn(3)
    X = X.values
    for i in range(10000):
        item = np.random.randint(0, X.shape[0])
        theta2 -= learning_rate * (np.dot(X[item], theta2.T) - y_train[item]) * X[item] + learning_rate *lamda *theta2
            # + learning_rate *lamda *theta2  is regularization term
            #   θ=θ-α(∑(h(x_i) − y_i))x_i-αλθ
    print('2',theta2)  # [2.9987092  4.00238675 4.98372683]


    # mini batch gradient decent
    batch = 20
    theta3 = np.random.randn(3)*0.1
    for i in range(1000):
        c =np.hstack([X, y_train.reshape(X.shape[0],-1)])
        np.random.shuffle(c)
        X_s = c[:,:-1]
        y_s=c[:,-1]
        for j in range(batch):  # 20 batches
            size = X_s.shape[0]//batch
            X_b = X_s[j*size: (j+1)*size]
            y_b = y_s[j*X.shape[0]//batch: (j+1)*X.shape[0]//batch]
            theta3 -= (learning_rate/X_b.shape[0]) * np.dot((np.dot(X_b, theta3.T) - y_b).T,X_b) \
                      + learning_rate/X_b.shape[0] * lamda * theta3
            # θ′=θ−α/n (∑_n(h(x_i) − y_i))x_i −α/n λθ
    print('3', theta3)    # [2.99919959 3.99806113 5.01670009]


def logistic_regression():
    np.random.seed(8)
    X, y = make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=1.5, random_state=6)
    #plt.scatter(X[:,0], X[:,1], c=y)
    #plt.show()
    X = np.hstack((X, np.ones(len(X)).reshape(len(X),-1)))
    y = y.reshape(len(X), -1)
    theta = np.random.randn(1, 3) * 0.1
    batch = 10
    lr = 0.01
    cost_list = []
    for i in range(10000):
        # shuffle data
        c = np.hstack([X, y])
        np.random.shuffle(c)
        X_s = c[:, :-1]
        y_s = c[:, -1]

        cost = 0 # cost
        for j in range(batch):  # 20 batches
            size = X_s.shape[0]//batch
            X_b = X_s[j*size: (j+1)*size]
            y_b = y_s[j*size: (j+1)*size].reshape(-1,1)
            h = 1 / (1 + np.exp(-1.0*(np.dot(X_b, theta.T))))   # prediction
            cost += np.sum(-y_b*np.log(h) - (1 - y_b)* np.log(1 - h))
            d_theta = np.dot((h-y_b).T, X_b) / len(X_b)  # gradient
            theta -= lr * d_theta
        cost_list.append(cost)
        #print(error)
    print(theta)

    '''
    lr = LogisticRegression(max_iter=10000)  # solver='newton-cg',max_iter=100
    lr.fit(X, y)
    coef = lr.coef_
    print(coef)
    print(lr.score(X,y))'''

    X_test, y_test = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=1.5, random_state=6)
    X_test = np.hstack((X_test, np.ones(len(X_test)).reshape(len(X_test), -1)))
    y_test = y_test.reshape(len(X_test), -1)
    h = 1 / (1 + np.exp(-1.0 * (np.dot(X_test, theta.T))))    # predict test data
    correct = 0
    for i in range(len(h)):
        pred = 0
        if h[i][0] > 0.5:
            pred = 1
        if y_test[i] == pred:
            correct += 1
    print('score: ',correct / len(h))

    a = np.linspace(2,12,200)
   # plt.plot(a, (0-coef[0,2]-a*coef[0,0])/coef[0,1])
    plt.plot(a, (0 - theta[0, 2] - a * theta[0, 0]) / theta[0, 1])
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    plt.plot(cost_list)
    plt.xscale('log')
    plt.xlim(xmin=1)
    plt.show()


def decision_tree():
    size = 1000
    theta = np.random.rand(size) * 2.0 * np.pi
    x, y = np.cos(theta), np.sin(theta)
    #y += np.random.randn(size)*0.1
    #x += np.random.randn(size) * 0.1
    y[::4] += np.random.randn(size//4)*0.3
    plt.scatter(x, y)
    plt.axis('equal')
    plt.show()


    tree = DecisionTreeRegressor(max_depth=6)
    tree.fit(theta.reshape(-1,1), np.c_[x,y])

    x_test = np.linspace(0, 2*np.pi, 100).reshape(-1,1)
    y_test = tree.predict(x_test)
    print(tree.score(x_test,y_test))


    rfc = RandomForestRegressor(max_depth=4)
    rfc.fit(np.hstack((theta.reshape(-1, 1),(np.random.randn(size,1)*0.5))), np.c_[x, y])
        # add a noise column
    x_test2 = np.hstack((np.linspace(0, 2 * np.pi, 100).reshape(-1, 1),(np.random.randn(100,1)*0.5)))
    y_test2 = rfc.predict(x_test2)
    print(rfc.score(x_test2, y_test2))
    print(rfc.feature_importances_)   # show importance of each column feature

    # plot test result
    fig = plt.figure(figsize=(2 * 6, 1 * 6))  # 6*5 size each for 2 subplots
    fig.suptitle("Decision Tree vs Random Forest", fontsize=16)
    axes1 = plt.subplot(1, 2, 1)  # return .axes.SubplotBase class

    axes1.scatter(y_test[:,0], y_test[:,1])
    axes1.axis('equal')
    axes1.set_title('Decision Tree', fontsize=16)

    axes2 = plt.subplot(1, 2, 2)
    axes2.scatter(y_test2[:, 0], y_test2[:, 1])
    axes2.axis('equal')
    axes2.set_title('Random Forest', fontsize=16)

    plt.show()

    data = load_iris()
    data, target = data['data'][:, 0:2], data['target']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)
    tree = DecisionTreeRegressor(max_depth=2)
    tree.fit(X_train, y_train)

    x, y = np.linspace(data[:, 0].min(), data[:, 0].max(), 1000), np.linspace(data[:, 1].min(), data[:, 1].max(), 1000)
    X, Y = np.meshgrid(x, y)
    XY = np.c_[X.ravel(), Y.ravel()]
    y_ = tree.predict(XY)
    plt.pcolormesh(X, Y, y_.reshape(1000, 1000))

    plt.scatter(data[:, 0], data[:, 1], c=target, cmap='rainbow')
    plt.show()

    print(tree.score(X_test, y_test))
    print(tree.feature_importances_)

def naive_bayes():
    data = load_iris()
    data, target = data['data'][:, 0:2], data['target']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)

    estimators = {'gaussian': GaussianNB(), 'multinomial': MultinomialNB(), 'bernoulli': BernoulliNB()}
    for k, est in estimators.items():
        est.fit(X_train, y_train)
        print(est.score(X_train, y_train), est.score(X_test, y_test))

        x, y = np.linspace(data[:, 0].min(), data[:, 0].max(), 1000), np.linspace(data[:, 1].min(), data[:, 1].max(), 1000)
        X, Y = np.meshgrid(x, y)
        XY = np.c_[X.ravel(), Y.ravel()]
        y_ = est.predict(XY)
        plt.pcolormesh(X, Y, y_.reshape(1000, 1000))

        plt.scatter(data[:, 0], data[:, 1], c=target, cmap='rainbow')
        plt.show()



    # spam sms detect
    sms = pd.read_csv('../resources/data/spam.csv', sep=',', encoding="ISO-8859-1")
    sms = sms.iloc[1:, 0:2].reset_index(drop=True)
    X_train, X_test = sms.iloc[:5000, 1], sms.iloc[5000:, 1]
    y_train, y_test = sms.iloc[:5000, 0], sms.iloc[5000:, 0]

    # tokenize sentence
    tf = TfidfVectorizer()  # convert sentence to a long array of words features with value 0,1
    # tf.fit(data)  # tf_data=tf.transform(data).toarray()   # same as fit_transform
    X_train = tf.fit_transform(X_train).toarray()  # convert to numpy array (5000*8220) , mostly 0
    X_test = tf.transform(X_test).toarray()

    # predict data
    x, y = sms.iloc[-20:, 1], sms.iloc[-20:, 0]
    x_tf = tf.transform(x).toarray()
    print(x)
    estimators = {'gaussian': GaussianNB(), 'multinomial': MultinomialNB(), 'bernoulli': BernoulliNB()}
    for k, est in estimators.items():
        est.fit(X_train, y_train)
        print(est.score(X_train, y_train), est.score(X_test, y_test))

        y_ = est.predict(x_tf)
        print(y.tolist(), y_)

def svm():
    np.random.seed(8)
    data, target = make_blobs(n_samples=200, centers=2, random_state=2)

    svc = SVC(kernel='linear', C=0.1)   # default C=1, smaller C stronger regularization
    svc.fit(data, target)
    # z = w1*x1 + w2*x2 + b   on the slicing plane(line): 0 = w1*x + w2*y + b
    w1, w2 = svc.coef_[0, 0], svc.coef_[0, 1]
    b = svc.intercept_
    x = np.linspace(-4,3,100)
    y = -w1/w2 * x - b/w2
    plt.scatter(data[:,0], data[:,1], c=target)
    plt.plot(x, y, c='r')

    #print(svc.support_vectors_)
    # [[-1.5841884  -6.8961805 ][-0.72864791 -7.18926735][ 1.10320057 -3.20707537][-0.32431771 -3.31914574]]
    b_up = svc.support_vectors_[2][1] - (-w1/w2) * svc.support_vectors_[2][0]
    b_down = svc.support_vectors_[1][1] - (-w1 / w2) * svc.support_vectors_[1][0]
    plt.plot(x, (-w1 / w2) * x + b_up, ls='--', c='g')
    plt.plot(x, (-w1 / w2) * x + b_down, ls='--', c='k')
    plt.scatter(svc.support_vectors_[:,0], svc.support_vectors_[:,1], c=(1,2,3,4), s=200, alpha=0.4, cmap='rainbow' )
    plt.show()


    data = np.random.randn(200,2)
    target = np.logical_xor(data[:, 0] > 0, data[:, 1] > 0)  #
    svc = SVC(kernel='rbf')
    grid = {'C': [0.1, 1, 5, 10], 'gamma': [0.025, 0.25, 2.5]}
    clf = GridSearchCV(estimator=svc, param_grid=grid, cv=5, n_jobs=2)
    clf.fit(data, target)
    print(clf.best_score_, clf.best_params_, clf.best_estimator_)
    # clf.predict
    x = np.linspace(data[:, 0].min(), data[:, 0].max(), 1000)
    y = np.linspace(data[:, 1].min(), data[:, 1].max(), 1000)
    X, Y = np.meshgrid(x, y)
    XY = np.c_[X.ravel(), Y.ravel()]
    distance = clf.decision_function(XY)   # distance to decision plane
    plt.figure(figsize=(6,6))
    plt.imshow(distance.reshape(1000,1000), extent=[data[:, 0].min(),data[:, 0].max(),data[:, 1].min(), data[:, 1].max()], cmap='PuOr_r')
    plt.contour(X, Y, distance.reshape(1000,1000))
    plt.scatter(data[:, 0], data[:, 1], c=target)
    plt.show()


def kmeans():
    np.random.seed(8)

    data, target = make_blobs(n_samples=200, centers=5, random_state=3)
    scores = []
    plt.figure()
    for k in range(2, 10):
        kmeans = KMeans(k)
        kmeans.fit(data)
        centers = kmeans.cluster_centers_
        axes = plt.subplot(4, 2, k-1)
        axes.axis('equal')
        axes.scatter(data[:, 0], data[:, 1], c=kmeans.labels_)
        axes.scatter(centers[:, 0], centers[:, 1], c=[i for i in range(k)], s=300, alpha=0.5, cmap='rainbow')
        axes.set_xlim([-15, 15])
        scores.append(silhouette_score(data, kmeans.labels_))
        # sihouette_samples(data, kmeans.labels_)  return all data silhouette_score
        # sihouette_samples(data, kmeans.labels_).mean() == silhouette_score(data, kmeans.labels_)
    plt.show()

    plt.plot(range(2,10), scores)
    plt.grid()
    plt.xticks(np.arange(2, 10, 1))
    plt.show()



    image = load_sample_image(image_name='china.jpg')
    data = np.array(image.reshape(-1, 3))
    # data_shuffled = data.copy()
    # np.random.shuffle(data_shuffled)
    data_shuffled = pd.DataFrame(data).sample(frac=0.2).values
    n_clusters = [8, 16, 32, 64, 128]

    plt.figure(figsize=(10 * 2, 8 * 3))  # 2*3 plots   10*8 size
    axes1 = plt.subplot(2, 3, 1)
    axes1.imshow(image)

    for i in range(len(n_clusters)):
        clf = KMeans(n_clusters=n_clusters[i])
        clf.fit(data_shuffled)  # clf.fit(data_shuffled[:1000])   # train on sampling pixcels
        main_colors = clf.cluster_centers_
        labels = clf.predict(data)
        # new_image = np.array(np.floor(main_colors[labels]).reshape(*image.shape), dtype=np.uint8)
        new_image = np.full_like(image, 1, dtype=np.uint8)
        count = 0
        for j in range(image.shape[0]):
            for k in range(image.shape[1]):
                color = main_colors[labels[count]]
                new_image[j, k] = color
                count += 1

        axes = plt.subplot(2, 3, i + 2)
        axes.imshow(new_image)

        plt.imsave('../resources/images/china%s.png' % i, new_image / 255)  # 144,219, 300, 367kb size
        # plt.imsave('../resources/images/china%s.jpg' % i, new_image)
        # jpg size decrease ignorable since already compressed

    plt.show()


def score():
    np.random.seed(8)
    iris = load_iris()
    data, target = iris['data'], iris['target']
    x, y = data[target != 2], target[target != 2]  # binary classification

    df = pd.DataFrame(data)
    c1_0 = df.iloc[target==0, 0]  # get first column values with class=0
    c1_1 = df.iloc[target == 1, 0]
    c1_0.plot(kind='hist', density=True, bins=50, alpha=0.5)  # blue hist
    c1_1.plot(kind='hist', density=True, bins=100, alpha=0.5)  # orange hist
    plt.show()

    print(c1_0)
    X = np.hstack((x, np.random.randn(100, 800)))  # add 800 columns of noise data for 100 rows
    tprs = []
    aucs = []
    i = 0
    sf = StratifiedKFold(5)  # , random_state=8
    fpr_mean = np.linspace(0,1,100)  # get average roc curve fpr_mean as x axis [0,1)
    for train, test in sf.split(X, y):
        clf = LogisticRegression()
        #clf = SVC(probability=True)
        clf.fit(X[train], y[train])
        y_ = clf.predict_proba(X[test])
        y_pred = clf.predict(X[test])
        fpr, tpr, thresh = roc_curve(y[test], y_[:,1])
        f = interp1d(fpr, tpr)    #
        tpr_mean = f(fpr_mean)

        tprs.append(tpr_mean)
        auc_ = auc(fpr, tpr)
        aucs.append(auc_)
        i += 1
        plt.plot(fpr, tpr, label=f'fold {i}, auc: %.2f' % auc_, alpha=0.5)
        print(accuracy_score(y[test], y_pred),
            precision_score(y[test], y_pred),
            recall_score(y[test], y_pred),
            f1_score(y[test], y_pred),
            confusion_matrix(y[test], y_pred))
    tprs = np.array(tprs)
    tpr_mean = tprs.mean(axis=0)
    tpr_mean[0],tpr_mean[-1] = 0,1

    auc_mean = auc(fpr_mean, tpr_mean)   # get average auc from average fpr and tpr
    auc_std = np.array(aucs).std()
    plt.plot(fpr_mean, tpr_mean, label='auc mean: %.2f$\pm%.2f$' % (auc_mean,auc_std), c='g')

    print(tprs.shape)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #classifier_basic()
    #regressor_basic()
    #cifar10()
    #linear_regression()
    #logistic_regression()
    #decision_tree()
    #naive_bayes()
    #svm()
    #kmeans()
    score()
