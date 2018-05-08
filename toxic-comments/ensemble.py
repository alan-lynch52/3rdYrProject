from toxic_comments import *
def main():
    #load train and test data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    #assign x and y
    y = train.iloc[:,2:]
    LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    x = train.drop(labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate'],axis=1)
    #fill any missing data
    x['comment_text'].fillna("unknown", inplace=True)
    #drop id column
    x = x.drop(labels = ['id'],axis = 1)
    #split test into feature set and set of ids(needed for submission)
    test_ids = test.drop(labels = ['comment_text'], axis = 1)
    test = test.drop(labels = ['id'],axis= 1)
    test['comment_text'].fillna("unknown", inplace=True)
    del train
    #concatenate traing and test data for bag of word extraction
    df = pd.concat([x['comment_text'], test['comment_text']], axis=0)
    
    tfidf_vec = TfidfVectorizer(max_features = 100000)
    tfidf_vec.fit(df)
    #transform train and test into into tfidf feature set
    train_tfidf = tfidf_vec.transform(x['comment_text'])
    test_tfidf = tfidf_vec.transform(test['comment_text'])

    #perform model tuning
    clf = LogisticRegression(solver='sag',C=0.5, tol=0.01)
##    n_estimators = [10,25,50,100]
##    for n in n_estimators:
##        bag_lr = BaggingClassifier(base_estimator=clf, n_estimators=n)
##        benchmark(str(n),bag_lr, train_tfidf, y)
    #OPTIMAL BAGGING N = 25
##    for n in n_estimators:
##        ada_lr = ExtraTreesClassifier(n_estimators=n)
##        benchmark('ET-'+str(n),ada_lr, train_tfidf, y)
    #ALL N ACHIEVED SAME ACCURACY, N=10
##    for n in n_estimators:
##        gb = GradientBoostingClassifier(n_estimators=n)
##        benchmark('GB-'+str(n),gb,train_tfidf,y)
    for n in n_estimators:
        rf = RandomForestClassifier(n_estimators=n)
        benchmark('RF-'+str(n),rf,train_tfidf,y)

    #Ensemble Experiment
    et = ExtraTreesClassifier(n_estimators=10)
    bag_lr = BaggingClassifier(base_estimator=LogisticRegression(solver='sag',C=0.5, tol=0.01), n_estimators=25)
    rf = RandomForestClassifier(n_estimators=15)
    gb = GradientBoostingClassifier(n_estimators=10)
    d = collections.OrderedDict()
    d['ET'] = et
    d['Bagging'] = bag_lr
    d['RF'] = rf
    d['GB'] = gb
##    get_auroc(d,train_tfidf,y)
##    get_balanced_accuracy(d,train_tfidf, y)
##    plot_cm(d,train_tfidf, y)

    #probabilities used for submission at Kaggle
    get_probability(train_tfidf, y, test_tfidf, test_ids, model=et).to_csv('et-submission.csv',index=False)
    get_probability(train_tfidf, y, test_tfidf, test_ids, model=bag_lr).to_csv('bag-submission.csv',index=False)
    get_probability(train_tfidf, y, test_tfidf, test_ids, model=rf).to_csv('rf-submission.csv',index=False)
    get_probability(train_tfidf, y, test_tfidf, test_ids, model=gb).to_csv('gb-submission.csv',index=False)
if __name__ == '__main__':
    main()
