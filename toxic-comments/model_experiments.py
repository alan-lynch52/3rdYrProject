from toxic_comments import *
def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    ##import warnings
    ##warnings.filterwarnings("ignore")
    y = train.iloc[:,2:]
    LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    x = train.drop(labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate'],axis=1)
    x['comment_text'].fillna("unknown", inplace=True)
    x = x.drop(labels = ['id'],axis = 1)
    test_ids = test.drop(labels = ['comment_text'], axis = 1)
    test = test.drop(labels = ['id'],axis= 1)
    test['comment_text'].fillna("unknown", inplace=True)
    del train

    df = pd.concat([x['comment_text'], test['comment_text']], axis=0)
    #FEATURE EXTRACTION EXPERIMENTS

    #GET BINARY FEATURE
    binary_vec = TfidfVectorizer(use_idf = False,norm = None, binary = True)
    binary_vec.fit(df)
    bin_train = binary_vec.transform(x['comment_text'])
    bin_test = binary_vec.transform(test['comment_text'])

    #GET TFIDF FEATURE
    tfidf_vec = TfidfVectorizer(max_features=20000)
    tfidf_vec.fit(df)
    train_tfidf = tfidf_vec.transform(x['comment_text'])
    test_tfidf = tfidf_vec.transform(test['comment_text'])



    ###Random Forest hyper parameter tuning
    ##rfc = RandomForestClassifier(n_estimators=10)
    ##rfc_benchmarks = benchmark('rfc-nestimators-10',rfc, train_tfidf, y)
    ###write_dict_to_csv(rfc_benchmarks, 'model-benchmarks.csv')
    ##rfc = RandomForestClassifier(n_estimators=15)
    ##rfc_benchmarks = benchmark('rfc-nestimators-15',rfc, train_tfidf, y)
    ###write_dict_to_csv(rfc_benchmarks, 'model-benchmarks.csv')
    ##rfc = RandomForestClassifier(n_estimators=25)
    ##rfc_benchmarks = benchmark('rfc-nestimators-25',rfc, train_tfidf, y)
    ###write_dict_to_csv(rfc_benchmarks, 'model-benchmarks.csv')
    ##
    ###Logistic Regression Hyper parameter tuning
    ##lr_ncg = LogisticRegression(solver='newton-cg')
    ##lr_lbfgs = LogisticRegression(solver='lbfgs')
    ##lr_liblinear = LogisticRegression(solver='liblinear')
    ##lr_sag = LogisticRegression(solver='sag')
    ##lr_saga = LogisticRegression(solver='saga')
    ###tuning solver
    ##ncg_benchmark = benchmark("lr-solver-ncg",lr_ncg, train_tfidf, y)
    ##write_dict_to_csv(ncg_benchmark, 'model-benchmarks.csv')
    ##lbfgs_benchmark = benchmark("lr-solver-lbfgs",lr_lbfgs, train_tfidf, y)
    ##write_dict_to_csv(lbfgs_benchmark, 'model-benchmarks.csv')
    ##liblinear_benchmark = benchmark("lr-solver-liblinear",lr_liblinear, train_tfidf, y)
    ##write_dict_to_csv(liblinear_benchmark, 'model-benchmarks.csv')
    ##sag_benchmark = benchmark("lr-solver-sag",lr_sag, train_tfidf, y)
    ##write_dict_to_csv(sag_benchmark, 'model-benchmarks.csv')
    ##saga_benchmark = benchmark("lr-solver-saga",lr_saga, train_tfidf, y)
    ##write_dict_to_csv(saga_benchmark, 'model-benchmarks.csv')
    ###All attained 3-fold CV accuracy of 0.9798, so sag was chosen as optimal as it took the least time to complete
    ##
    ###tuning C
    ##lr_c1 = LogisticRegression(solver='sag', C=0.1)
    ##c1_benchmark=benchmark("lr-c-0.1",lr_c1,train_tfidf, y)
    ##write_dict_to_csv(c1_benchmark, 'model-benchmarks.csv')
    ##
    ##lr_c2= LogisticRegression(solver='sag', C=0.2)
    ##c2_benchmark=benchmark("lr-c-0.2",lr_c2,train_tfidf, y)
    ##write_dict_to_csv(c2_benchmark, 'model-benchmarks.csv')
    ##
    ##lr_c3= LogisticRegression(solver='sag', C=0.3)
    ##c3_benchmark=benchmark("lr-c-0.3",lr_c3,train_tfidf, y)
    ##write_dict_to_csv(c3_benchmark, 'model-benchmarks.csv')
    ##
    ##lr_c4= LogisticRegression(solver='sag', C=0.4)
    ##c4_benchmark=benchmark("lr-c-0.4",lr_c4,train_tfidf, y)
    ##write_dict_to_csv(c4_benchmark, 'model-benchmarks.csv')
    ##
    ##lr_c5= LogisticRegression(solver='sag', C=0.5)
    ##c5_benchmark=benchmark("lr-c-0.5",lr_c5,train_tfidf, y)
    ##write_dict_to_csv(c5_benchmark, 'model-benchmarks.csv')
    ###highest scoring 3-fold CV accuracy: 0.9782 was attained with C=0.5
    ##
    ###tuning tol
    ##lr_tol01 = LogisticRegression(solver='sag',C=0.5, tol=0.01)
    ##tol01_benchmark = benchmark("lr-tol-0.01", lr_tol01, train_tfidf, y)
    ##write_dict_to_csv(tol01_benchmark, 'model-benchmarks.csv')
    ##
    ##lr_tol001= LogisticRegression(solver='sag',C=0.5, tol=0.001)
    ##tol001_benchmark = benchmark("lr-tol-0.001", lr_tol001, train_tfidf, y)
    ##write_dict_to_csv(tol001_benchmark, 'model-benchmarks.csv')
    ##
    ##lr_tol0001= LogisticRegression(solver='sag',C=0.5, tol=0.0001)
    ##tol0001_benchmark = benchmark("lr-tol-0.0001", lr_tol0001, train_tfidf, y)
    ##write_dict_to_csv(tol0001_benchmark, 'model-benchmarks.csv')
    ##
    ##lr_tol00001= LogisticRegression(solver='sag',C=0.5, tol=0.00001)
    ##tol00001_benchmark = benchmark("lr-tol-0.00001", lr_tol00001, train_tfidf, y)
    ##write_dict_to_csv(tol00001_benchmark, 'model-benchmarks.csv')
    ###all tolerances attained 3-fold cv accuracy of: 0.9782, there a tol of 0.01 will be chosen as it took the least amount of time
    ##
    ###Bernoulli NB Hyper-parameter tuning
    ###tuning alpha
    ##bnb9 = BernoulliNB(alpha=0.9)
    ##bnb9_benchmarks = benchmark("bernoulliNB-alpha-0.9",bnb9, bin_train, y)
    ##write_dict_to_csv(bnb9_benchmarks, 'model-benchmarks.csv')
    ##
    ##bnb8 = BernoulliNB(alpha=0.8)
    ##bnb8_benchmarks = benchmark("bernoulliNB-alpha-0.8",bnb8, bin_train, y)
    ##write_dict_to_csv(bnb8_benchmarks, 'model-benchmarks.csv')
    ##
    ##bnb7= BernoulliNB(alpha=0.7)
    ##bnb7_benchmarks = benchmark("bernoulliNB-alpha-0.7",bnb7, bin_train, y)
    ##write_dict_to_csv(bnb7_benchmarks, 'model-benchmarks.csv')
    ##
    ##bnb6 = BernoulliNB(alpha=0.6)
    ##bnb6_benchmarks = benchmark("bernoulliNB-alpha-0.6",bnb6, bin_train, y)
    ##write_dict_to_csv(bnb6_benchmarks, 'model-benchmarks.csv')
    ##
    ##bnb5 = BernoulliNB(alpha=0.5)
    ##bnb5_benchmarks = benchmark("bernoulliNB-alpha-0.5",bnb5, bin_train, y)
    ##write_dict_to_csv(bnb5_benchmarks, 'model-benchmarks.csv')
    ###Highest 3-fold cv accuracy: 0.9695 was achieved by alpha = 0.6
    ##
    ###Multinomial NB Hyper-parameter tuning
    ###tuning alpha
    ##mnb1 = MultinomialNB(alpha=1.0)
    ##mnb1_benchmarks = benchmark("multinomialNB-alpha-1.0",mnb1, train_tfidf, y)
    ##write_dict_to_csv(mnb1_benchmarks, 'model-benchmarks.csv')
    ##
    ##mnb9 = MultinomialNB(alpha=0.9)
    ##mnb9_benchmarks = benchmark("multinomialNB-alpha-0.9",mnb9, train_tfidf, y)
    ##write_dict_to_csv(mnb9_benchmarks, 'model-benchmarks.csv')
    ##
    ##mnb8 = MultinomialNB(alpha=0.8)
    ##mnb8_benchmarks = benchmark("multinomialNB-alpha-0.8",mnb8, train_tfidf, y)
    ##write_dict_to_csv(mnb8_benchmarks, 'model-benchmarks.csv')
    ##
    ##mnb7 = MultinomialNB(alpha=0.7)
    ##mnb7_benchmarks = benchmark("multinomialNB-alpha-0.7",mnb7, train_tfidf, y)
    ##write_dict_to_csv(mnb7_benchmarks, 'model-benchmarks.csv')
    ##
    ##mnb6 = MultinomialNB(alpha=0.6)
    ##mnb6_benchmarks = benchmark("multinomialNB-alpha-0.6",mnb6, train_tfidf, y)
    ##write_dict_to_csv(mnb6_benchmarks, 'model-benchmarks.csv')
    ##
    ##mnb5 = MultinomialNB(alpha=0.5)
    ##mnb5_benchmarks = benchmark("multinomialNB-alpha-0.5",mnb5, train_tfidf, y)
    ##write_dict_to_csv(mnb5_benchmarks, 'model-benchmarks.csv')
    ##
    ##mnb4 = MultinomialNB(alpha=0.4)
    ##mnb4_benchmarks = benchmark("multinomialNB-alpha-0.4",mnb4, train_tfidf, y)
    ##write_dict_to_csv(mnb4_benchmarks, 'model-benchmarks.csv')
    ##
    ##mnb3 = MultinomialNB(alpha=0.3)
    ##mnb3_benchmarks = benchmark("multinomialNB-alpha-0.3",mnb3, train_tfidf, y)
    ##write_dict_to_csv(mnb3_benchmarks, 'model-benchmarks.csv')
    ##
    ##mnb2 = MultinomialNB(alpha=0.2)
    ##mnb2_benchmarks = benchmark("multinomialNB-alpha-0.2",mnb2, train_tfidf, y)
    ##write_dict_to_csv(mnb2_benchmarks, 'model-benchmarks.csv')
    ##
    ##mnb1 = MultinomialNB(alpha=0.1)
    ##mnb1_benchmarks = benchmark("multinomialNB-alpha-0.1",mnb1, train_tfidf, y)
    ##write_dict_to_csv(mnb1_benchmarks, 'model-benchmarks.csv')
    ##
    ##mnb01 = MultinomialNB(alpha=0.01)
    ##mnb01_benchmarks = benchmark("multinomialNB-alpha-0.01",mnb01, train_tfidf, y)
    ##write_dict_to_csv(mnb01_benchmarks, 'model-benchmarks.csv')
    #highest 3-fold cv attained by alpah of 0.1 and 0.01, 0.1 took less time and will be used

    c_list = [0.01,0.1,0.25,0.5,0.75,1.0]

    for c in c_list:
        svm = SVC(C=c,kernel='linear',probability=True)
        benchmark('svc-'+str(c),svm,train_tfidf, y)
        
    #optimal models found
##    optimal_lr = LogisticRegression(solver='sag',C=0.5, tol=0.01)    
##    optimal_bnb = BernoulliNB(alpha=0.6)
##    optimal_mnb = MultinomialNB(alpha=0.1)
##    optimal_svc = LinearSVC(C=0.5)
##    d = collections.OrderedDict()
##    d['LR'] = optimal_lr
##    d['MNB'] = optimal_mnb
##    d['BNB'] = optimal_bnb
##    d['svc'] = optimal_svc
##
##
##    get_auroc(d, train_tfidf, y)
##    plot_cm(d,train_tfidf, y)
##    get_balanced_accuracy(d,train_tfidf, y)

    ###benchmark optimal models
    ##lr_benchmarks = benchmark('lr-solver-sag-c-0.5-tol-0.01',optimal_lr, train_tfidf, y)
    ##write_dict_to_csv(lr_benchmarks,'model-benchmarks.csv')
    ##
    ##bnb_benchmarks = benchmark('bnb-alpha-0.6',optimal_bnb, train_tfidf, y)
    ##write_dict_to_csv(bnb_benchmarks,'model-benchmarks.csv')
    ##
    ##mnb_benchmarks = benchmark('mnb-alpha-0.1', optimal_mnb, train_tfidf, y)
    ##write_dict_to_csv(mnb_benchmarks,'model-benchmarks.csv')
    ##
    ##rfc_benchmarks = benchmark('rfc-nestimators-15', optimal_rfc, train_tfidf, y)
    ##write_dict_to_csv(rfc_benchmarks,'model-benchmarks.csv')
    ##
    ###predictions
    ##lr_preds = make_prediction(train_tfidf, y, test_tfidf, test_ids, model=optimal_lr)
    ##lr_preds.to_csv('lr_submission.csv',index=False)
    ###0.9696 LB attained on optimal LogisticRegression
    ##
    ##rfc_preds = make_prediction(train_tfidf, y, test_tfidf, test_ids, model=optimal_rfc)
    ##rfc_preds.to_csv('rfc_submission.csv',index=False)
    ###0.8785 LB attain on optimal Random Forest
    ##bnb_preds = make_prediction(bin_train, y, bin_test, test_ids, model=optimal_bnb)
    ##bnb_preds.to_csv('bnb_submission.csv',index=False)
    ###0.8804 LB attained on optimal bernoulli NB
    ##mnb_preds = make_prediction(train_tfidf, y, test_tfidf, test_ids, model=optimal_mnb)
    ##mnb_preds.to_csv('mnb_submission.csv',index=False)
    ###0.8708 LB attained on optimal multinomial NB
if __name__ == '__main__':
    main()
