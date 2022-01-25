'''
    current fraud detection expert rule based + supervised learning: gradient boost decision tree(GBDT)

    recent paper regarding deep learning(LSTM) and unsupervised learning(isolation forest)
        read data from s3, feature engineering: feature conversion(time column), feature selection(discard useless
        feature by checking histogram of each feature on fraud and non fraud class distribution), feature scaling:
        (standard scaler), use gbdt for feature importance ranking, smote (generate imbalance data less class data), use
        tensorflow lstm and sklearn isolation forest model, find recall and f1 score, use randomizedsearchcv for
        parameter tuning (tensorflow use keras scikit learn wrappers to wrap tf model and use randomizedsearchcv)
        together with rule based only see around 1-2% increase (95->96) in f1 and recall score on validation set



    I graduated with my master degree in Computer and Information Science in 2019. I chose that field of study because
        I’ve believe that computer science is one of the most useful subject in the future and I love to surf on
        the latest technology trend. After graduation, I joined Capital one as a software engineer in a feature
        development team, we've accomplished onboarding several big retail company(walmart, neiman marcus) for their
        PLCC(Private Label Store Credit Card) and cobrand credit card, invloving add new feature file, different rules
        in designated repository and integrate with other existing component in the authentication decisioning framework.
        We also owned a referral decision database based on AWS dynamoDB, and corresponding api for querying.
        Recently we had a story with migrate to new managing pipeline, which heavily involves devop tools like jenkins
        docker and kubernetes
        On side project, I have worked on various web development, machine learning, crawling framework associate with
        various SQL, NoSQL database. I also have some practice on PySpark for processing large data.
          




    questions to ask
    how can I ready myself to be the most successful @ BP
    what do the daily team cadences(routine) look like?
    what kind of project does BP have currently for the data team?
    what 's the company's culture focusing on?
    Is there room for career growth as contractor?
    Is this role more about developing new stuff, discovering insights or maintaining the current owned product?
        what percentage time used on those
    what do you like most about the company?
    what are the biggest problems you are solving for today, and what is their impact on BP’s business

'''