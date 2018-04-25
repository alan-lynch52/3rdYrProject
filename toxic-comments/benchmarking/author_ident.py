#https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
import sys
sys.path.insert(0, '..')

from toxic_comments import *

def main():
    train = pd.read_csv('author_ident_train.csv')
    x = train['text']
    y = train['author']
    y = y.to_frame(name="author")
    y = pd.get_dummies(y)
    del train

    tfidf_vec = TfidfVectorizer()
    tfidf_train = tfidf_vec.fit_transform(x)

    count_vec = CountVectorizer()
    count_train = count_vec.fit_transform(x)

    bin_vec=  TfidfVectorizer(use_idf = False, binary=True, norm=False)
    bin_train = bin_vec.fit_transform(x)
    print(x.shape)
    print(y.shape)
    base_model = LogisticRegression()
    d = collections.OrderedDict()
    #FEATURE EXTRACTION
##    d['TFIDF'] = tfidf_train
##    d['Count'] = count_train
##    d['Binary'] = bin_train
##    benchmark('TFIDF',base_model, tfidf_train, y,cv=3)
##    benchmark('Count',base_model, count_train, y,cv=3)
##    benchmark('Binary',base_model, bin_train, y,cv=3)
##    get_auroc_fe(d,y)
##    get_balanced_accuracy_fe(d,y)
    #FEATURE STACKING
##    tfidf_train_char = TfidfVectorizer(analyzer='char').fit_transform(x)
##    d['TFIDF+Count+Bin'] = hstack([tfidf_train,count_train, bin_train])
##    d['TFIDF+Count'] = hstack([tfidf_train, count_train])
##    d['TFIDF+Bin'] = hstack([tfidf_train, bin_train])
##    d['TFIDF-word+char'] = hstack([tfidf_train,tfidf_train_char])
##    benchmark('TFIDF+Count+Bin',base_model, d['TFIDF+Count+Bin'], y, cv=3)
##    benchmark('TFIDF+Count', base_model, d['TFIDF+Count'],y,cv=3)
##    benchmark('TFIDF+Bin',base_model, d['TFIDF+Bin'],y,cv=3)
##    benchmark('TFIDF-word+char',base_model, d['TFIDF-word+char'], y, cv=3)
##    get_auroc_fe(d,y)
##    get_balanced_accuracy_fe(d,y)
    #FEATURE SELECTION
##    k = int(len(tfidf_vec.get_feature_names()) / 2)
##    kbest = SelectKBest(chi2, k=k)
##    rfe = RFE(base_model, step=0.05)
##    sfm = SelectFromModel(base_model)
##    d['kbest'] = kbest
##    d['rfe'] = rfe
##    d['sfm'] = sfm
##    benchmark('kbest',base_model, tfidf_train, y, fs=kbest)
##    benchmark('rfe',base_model, tfidf_train, y, fs=rfe)
##    benchmark('sfm',base_model, tfidf_train, y, fs=sfm)
##    get_auroc_fs(d, tfidf_train, y)
##    get_balanced_accuracy_fs(d, tfidf_train, y)
    #MODELING
##    lr = LogisticRegression(C=0.5, tol=0.01)
##    bnb = BernoulliNB(alpha = 1.0)
##    mnb = MultinomialNB(alpha = 1.0)
##    rf = RandomForestClassifier(n_estimators = 15)
##    d['lr'] = lr
##    d['bnb'] = bnb
##    d['mnb'] = mnb
##    d['rf'] = rf
##    
##    benchmark('lr', lr, tfidf_train, y)
##    benchmark('bnb', bnb, tfidf_train, y)
##    benchmark('mnb', mnb, tfidf_train, y)
##    benchmark('rf', rf, tfidf_train, y)
##    get_auroc(d, tfidf_train, y)
##    get_balanced_accuracy(d, tfidf_train, y)
##
##    x_train, x_val, y_train, y_val = train_test_split(tfidf_train, y, test_size=0.4, random_state = 2)
##    get_probability(x_train, y_train, x_val, model = lr).to_csv('auth_ident/lr-prob.csv',index=False)
##    get_probability(x_train, y_train, x_val, model = bnb).to_csv('auth_ident/bnb-prob.csv',index=False)
##    get_probability(x_train, y_train, x_val, model = mnb).to_csv('auth_ident/mnb-prob.csv',index=False)
##    get_probability(x_train, y_train, x_val, model = rf).to_csv('auth_ident/rf-prob.csv',index=False)
##    y_val.to_csv('true-labels.csv',index=False)
    #ENSEMBLING
##    lr_prob = pd.read_csv('auth_ident/lr-prob.csv')
##    bnb_prob = pd.read_csv('auth_ident/bnb-prob.csv')
##    mnb_prob = pd.read_csv('auth_ident/mnb-prob.csv')
##    rf_prob = pd.read_csv('auth_ident/rf-prob.csv')
##    
##    LABELS = ['author_EAP','author_HPL','author_MWS']
##    ens_prob = lr_prob.copy()
##    ens_prob[LABELS] = (lr_prob[LABELS] + bnb_prob[LABELS]) / 2
##    ens_prob.to_csv('auth_ident/e3.csv', index=False)

##    e1 = pd.read_csv('auth_ident/e1.csv')
##    e2 = pd.read_csv('auth_ident/e2.csv')
##    e3 = pd.read_csv('auth_ident/e3.csv')
##    e4 = pd.read_csv('auth_ident/e4.csv')
##    true = pd.read_csv('auth_ident/true-labels.csv')
##
##    d['e1'] = e1
##    d['e2'] = e2
##    d['e3'] = e3
##    d['e4'] = e4
##    get_auroc_ensemble(d, true)
##    get_balanced_accuracy_ensemble(d,true)

    et = ExtraTreesClassifier(n_estimators=10)
    bag_lr = BaggingClassifier(base_estimator=LogisticRegression(solver='sag',C=0.5, tol=0.01), n_estimators=25)
    rf = RandomForestClassifier(n_estimators=15)
    gb = GradientBoostingClassifier(n_estimators=10)
    d = collections.OrderedDict()
    d['ET'] = et
    d['Bagging'] = bag_lr
    d['RF'] = rf
    d['GB'] = gb
    benchmark('ET',et,tfidf_train,y)
    benchmark('Bagging',et,tfidf_train,y)
    benchmark('RF',rf,tfidf_train,y)
    benchmark('GB',gb,tfidf_train,y)
    get_auroc(d,tfidf_train,y)
    get_balanced_accuracy(d,tfidf_train, y)
    
if __name__ == '__main__':
    main()
