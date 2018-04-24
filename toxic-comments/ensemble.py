from toxic_comments import *
def round_dfs(d):
    for key in d:
        df = d[key]
        for label in df:
            df[label] = df[label].round()
    return d
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

    tfidf_vec = TfidfVectorizer(max_features = 100000)
    tfidf_vec.fit(df)
    train_tfidf = tfidf_vec.transform(x['comment_text'])
    test_tfidf = tfidf_vec.transform(test['comment_text'])
    ##lr_fp = 'val/models/lr_preds.csv'
    ##bnb_fp = 'val/models/bnb_preds.csv'
    ##mnb_fp = 'val/models/mnb_preds.csv'
    ##rfc_fp = 'val/models/rf_preds.csv'
    ##
    ##lr_preds = pd.read_csv(lr_fp)
    ##lr_probs = pd.read_csv('val/models/lr_probs.csv')
    ##
    ##bnb_preds = pd.read_csv(bnb_fp)
    ##bnb_probs = pd.read_csv('val/models/bnb_probs.csv')
    ##
    ##mnb_preds = pd.read_csv(mnb_fp)
    ##mnb_probs = pd.read_csv('val/models/mnb_probs.csv')
    ##
    ##rfc_preds = pd.read_csv(rfc_fp)
    ##rfc_probs = pd.read_csv('val/models/rf_probs.csv')
    ##
    LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    ##
    ##ensemble_preds = lr_preds.copy()
    ##ensemble_probs = lr_probs.copy()
    ##
    ##ensemble_preds[LABELS] = ((lr_preds[LABELS]*0.7) + (bnb_preds[LABELS]*0.1) + (mnb_preds[LABELS]*0.1) + (rfc_preds[LABELS]*0.1)) 
    ##ensemble_probs[LABELS] = ((lr_probs[LABELS]*0.7) + (bnb_probs[LABELS]*0.1) + (mnb_probs[LABELS]*0.1) + (rfc_probs[LABELS]*0.1))
    ##
    ##ensemble_preds.to_csv('val/ensembles/weighted_avg_preds.csv',index=False)
    ##ensemble_probs.to_csv('val/ensembles/weighted_avg_probs.csv',index=False)


    #get AUROC, CM, Bal Acc
    ##e1 = 'val/ensembles/lr_mnb_bnb_rf_probs.csv'
    ##e2 = 'val/ensembles/lr_mnb_bnb_probs.csv'
    ##e3 = 'val/ensembles/lr_bnb_probs.csv'
    ##e4 = 'val/ensembles/weighted_avg_probs.csv'
    ##true = 'val/ensembles/true_labels.csv'
    ##
    ##e1_probs = pd.read_csv(e1)
    ##e2_probs = pd.read_csv(e2)
    ##e3_probs = pd.read_csv(e3)
    ##e4_probs = pd.read_csv(e4)
    ##true_labels = pd.read_csv(true)
    ##d = collections.OrderedDict()
    ##d['e1'] = e1_probs
    ##d['e2'] = e2_probs
    ##d['e3'] = e3_probs
    ##d['e4'] = e4_probs

    clf = LogisticRegression(solver='sag',C=0.5, tol=0.01)
##    n_estimators = [10,25,50,100]
##    for n in n_estimators:
##        bag_lr = BaggingClassifier(base_estimator=clf, n_estimators=n)
##        benchmark(str(n),bag_lr, train_tfidf, y)
    #OPTIMAL BAGGING N = 25
##    for n in n_estimators:
##        ada_lr = AdaBoostClassifier(base_estimator=clf, n_estimators=n)
##        benchmark('AdaBoost-'+str(n),ada_lr, train_tfidf, y)
    #ALL N ACHIEVED SAME ACCURACY, N=10
##    for n in n_estimators:
##        gb = GradientBoostingClassifier(n_estimators=n)
##        benchmark('GB-'+str(n),gb,train_tfidf,y)

    ada_lr = AdaBoostClassifier(base_estimator=clf, n_estimators=25)
    bag_lr = BaggingClassifier(base_estimator=clf, n_estimators=25)
    rf = RandomForestClassifier(n_estimators=15)
    gb = GradientBoostingClassifier(n_estimators=10)
    d = collections.OrderedDict()
    d['adaboost'] = ada_lr
    d['bagging'] = bag_lr
    d['rf'] = rf
    d['gb'] = gb
    #get_auroc(d,train_tfidf,y)
    #get_balanced_accuracy(d,train_tfidf, y)
    #plot_cm_ensemble(d,true_labels)

    get_probability(train_tfidf, y, test_tfidf, test_ids, model=ada_lr).to_csv('ada-submission.csv',index=False)
    get_probability(train_tfidf, y, test_tfidf, test_ids, model=bag_lr).to_csv('bag-submission.csv',index=False)
    #get_probability(train_tfidf, y, test_tfidf, test_ids, model=rf).to_csv('rf-submission.csv',index=False)
    get_probability(train_tfidf, y, test_tfidf, test_ids, model=gb).to_csv('gb-submission.csv',index=False)
if __name__ == '__main__':
    main()
