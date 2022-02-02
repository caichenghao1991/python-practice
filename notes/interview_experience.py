'''
    current fraud detection expert rule based + supervised learning: gradient boost decision tree(GBDT)

    recent paper regarding deep learning(LSTM) and unsupervised learning(isolation forest)
        read data from s3, feature engineering: feature conversion(time column), feature selection(discard useless
        feature by checking histogram of each feature on fraud and non fraud class distribution), feature scaling:
        (standard scaler), use gbdt for feature importance ranking, smote (generate imbalance data less class data), use
        tensorflow lstm and sklearn isolation forest model, find recall and f1 score, use randomizedsearchcv for
        parameter tuning (tensorflow use keras scikit learn wrappers to wrap tf model and use randomizedsearchcv)
        together with rule based only see around 1-2% increase (95->96) in f1 and recall score on validation set



    I graduated with my master degree in Computer and Information Science in 2019. After graduation, I joined Capital
        one as a software engineer in a feature development team, we've accomplished onboarding several big retail
        company(walmart, neiman marcus) for their PLCC(Private Label Store Credit Card) and cobrand credit card。
        We also owned a referral decision database based on AWS dynamoDB, and corresponding api for querying.
        Recently we had a story with migrate to new managing pipeline, which heavily involves devop tools like jenkins
        docker and kubernetes
        On side project, I have worked on various web development, machine learning, crawling framework associate with
        various SQL, NoSQL database. I also have some practice on PySpark for processing large data.
        That's pretty much about myself


I chose that field of study because
        I’ve believe that computer science is one of the most useful subject in the future and I love to surf on
        the latest technology trend.
invloving add new feature file, different rules  in designated repository and integrate with other existing
        component in the authentication decisioning framework.


    questions to ask
    how can I ready myself to be the most successful
    what do the daily team cadences(routine) look like?
    what kind of project does BP have currently for the data team?
    Is there room for career growth?
    Is this role more about developing new stuff, discovering insights or maintaining the current owned product?
        what percentage time used on those

    what are the biggest problems you are solving for today, and what is their impact on BP’s business

    What are the biggest challenges of this job?
    What does a typical day look like?
    What are the most immediate projects that need to be addressed?

    What’s your favorite part about working here?
    How does the organization support your professional development and career growth?
    How would you describe a typical day in this position?

    What is the company's management style?
    How many teams are there in the IT department and how does team cooperate with each other?
    What are the company's plans for growth and development?
    Is the work environment here more collaborative or more independent?

    What’s the company and team culture like?
    What are the next steps in the interview process?
    Is there anything else I can provide you with that would be helpful?

    data management platform
    advertisement(publish, messaging)/marketing and customer info for analyzing
    Powered by CORE ID,® the most accurate and stable identity management platform representing 200+ million people,

    Strategic thinking. Creative vision. Collaborative spirit. Eagerness to grow and succeed.
     Technology/Skills to Target
    Spark, Scala, Python
    Hadoop, Hive, Spark, and related technologies
    Python,
    Airflow, Kubernetes, Docker
    data-driven solutions at scale

    why choose us
    I found many interesting posts from company on linkedin show casing the company's work and related clients. In tech
        aspect, Epsilon is utilizing many latest technology to deliver it's product. And I'd love to utilize those
        technology to achieve great result. And Epsilon value individual's opinion and have a flexible culture.
        This role suit my skill set and fit well in my ideal career growth path.

    handle stress
    stay positive, use stress as a motivator, focus on what I have control. make time for relaxation. stay focused on
        the subject and overcome difficulties one at a time. manage time according to importance and urgency matrix.


    Elastic Search
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