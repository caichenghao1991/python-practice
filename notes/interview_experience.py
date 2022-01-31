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


    Tableau/ Power BI
    Axioma Optimizer, Blackrock-Aladdin and Incorta
    MLFlow, Kubeflow
    multivariate
    data wrangling( process of cleaning and unifying messy and complex data sets for easy access and analysis),
    manipulation and management of technologies
    distributed data parallelism and distributed training
    Collaborative filtering
    Matrix factorization
    Ranking algorithms



    steps data scientist
    define goal, data collection.
    load data from logs
    sanity check ata
    scrape, munge, transform, clean data
    data exploration
    feature engineering
    feature selection
    choose model and fit mode
    generate model output
    model evaluation
    model result and visualization
    test hypothesis and data

    key technology: python, r, sql, jupyter notebooks, tensorflow, aws, unix, tableau, c, nosql, matlab, java, hadoop/
        hive/pig, spark/mllib, excel
    key knowledge: logistic regression, decision tree, random forest, neural networks, bayesian, ensemble methods, svm,
        gradient boost, cnn, rnn, evolutionary approaches, hmm, markov logic networks, gans
    questions: sql, algorithm, statistic (conditional probability, bayes rule, probability in sampling: bootstrap,
        reservoir, a/b test, p-value), probabilities and static inference(probabilities distribution(normal, binomial,
        poisson), law of large number, ventral limit theorem, expectation, variance, quantiles, correlation and
        covariance, parametric vs non parametric methods, bootstrap), hypothesis testing(causal inference, a/b
        experiment design, sample size calculation, type I, II error, power, p-value, one sample, two sample t-test,
        chi-square test, confidence interval, pitfalls, ANOVA), experimental design, machine learning (linear regression
        (simple/generalized/ multivariate linear regression, cost function, mse and mle(maximum likelihood estimation),
        hypothesis testing in lm, f test, multicollinearity, regularization(ridge, lasso, elastic net), bias variance
        tradeoff, cross validation), logistic regression(binomial/multinomial logistic regression, confusion matrix, roc
        auc, cost benefit analysis(profit curve)) svm, kernel, train decision tree(greedy, over fit), entropy, ensemble
        learning (bagging, random forest, gradient boost, adaboost), performance metrics, precision, recall, roc, auc,
        sampling impact performance(unbalance data), model selection(cross validation, hyper-parameter tuning, bias
        variance and over fitting, regularization and feature selection), neural network, auto encoder, back propagation,
        gradient vanish), natural language processing(extract feature from text, tokenization and stop words, stemming
        and lemmatization, bag of words and tf-idf, distance and similarity, topic modeling and sentiment analysis,
        naive bayes), unsupervised learning(clustering(kmeans, hierarchical clustering), dimension reduction(PCA, SVD,
        non-negative matrix factorization, latent features and relation with clustering)), deep learning(auto encoder,
        feed-forward neutal network, CNN, RNN, embedding space, word embedding, LSTM, transfer learning)
        case study (how to build a
        recommendation(cold start and exploration, collaborative filtering, content-based recommendation, matrix
        factorization)/search/fraud detection system/predict churn rate)/lend money violation rate, customer transfer
        rate/ credit risk prediction/ price forecasting/ ads click through rate/ conversion rate prediction(marketing/
        user acquisition/ face detection), case interview(problem formation(clarify input output, goal), feature extraction and label/
        target definition, ML end-to-end workflow, cold start and exploration, experiment)
    business insight from model coefficient or perform experiment and test


































































































'''