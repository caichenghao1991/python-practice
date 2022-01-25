'''
    current fraud detection expert rule based + supervised learning: gradient boost decision tree(GBDT)

    recent paper regarding deep learning(LSTM) and unsupervised learning(isolation forest)
        read data from s3, feature engineering: feature conversion(time column), feature selection(discard useless
        feature by checking histogram of each feature on fraud and non fraud class distribution), feature scaling:
        (standard scaler), use gbdt for feature importance ranking, smote (generate imbalance data less class data), use
        tensorflow lstm and sklearn isolation forest model, find recall and f1 score, use randomizedsearchcv for
        parameter tuning (tensorflow use keras scikit learn wrappers to wrap tf model and use randomizedsearchcv)
        together with rule based only see around 1-2% increase (95->96) in f1 and recall score on validation set



'''