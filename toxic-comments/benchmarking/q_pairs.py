#https://www.kaggle.com/c/quora-question-pairs
import sys
sys.path.insert(0,'..')
from toxic_comments import *
def main():
    train = pd.read_csv('q_pairs_train.csv')
    train = train.dropna(axis=0)
    
    q1 = train['question1']
    q2 = train['question2']
    y = train['is_duplicate'].to_frame(name="is_duplicate")
    del train
    #df = pd.concat([q1,q2], axis=1, keys = ['question1','question2'])
    tfidf_vec = TfidfVectorizer()
##    char_vec = TfidfVectorizer(analyzer='char')
##    count_vec = CountVectorizer()
##    bin_vec = TfidfVectorizer(use_idf=False, norm=None, binary=True)

    tfidf_q1 = tfidf_vec.fit_transform(q1)
    tfidf_q2 = tfidf_vec.fit_transform(q2)
##    count_q1 = count_vec.fit_transform(q1)
##    count_q2 = count_vec.fit_transform(q2)
##    bin_q1 = bin_vec.fit_transform(q1)
##    bin_q2 = bin_vec.fit_transform(q2)
##    char_q1 = char_vec.fit_transform(q1)
##    char_q2 = char_vec.fit_transform(q2)
##    base_model = LogisticRegression()
    d = collections.OrderedDict()
    #Feature Extraction
##    tfidf_train = hstack([tfidf_q1,tfidf_q2])
##    count_train = hstack([count_q1, count_q2])
##    bin_train = hstack([bin_q1, bin_q2])
##    char_train = hstack([char_q1,char_q2])

##    d['TFIDF'] = tfidf_train
##    d['Count'] = count_train
##    d['Binary'] = bin_train
##    benchmark('TFIDF',base_model,tfidf_train, y)
##    benchmark('Count', base_model, count_train, y)
##    benchmark('Binary', base_model, bin_train, y)
##    get_auroc_fe(d,y)
##    get_balanced_accuracy_fe(d,y)

    #Feature Stacking
    d['TFIDF-count-bin'] = hstack([tfidf_train,count_train,bin_train])
    d['TFIDF-count'] = hstack([tfidf_train, count_train])
    d['TFIDF-bin'] = hstack([tfidf_train, bin_train])
    d['TFIDF-word-char'] = hstack([tfidf_train, char_train])

    ##benchmark('TFIDF-count-bin', base_model, d['TFIDF-count-bin'],y)
    benchmark('TFIDF-count', base_model, d['TFIDF-count'], y)
    benchmark('TFIDF-bin', base_model, d['TFIDF-bin'], y)
    benchmark('TFIDF-word-char', base_model, d['TFIDF-word-char'],y)
    get_auroc_fe(d,y)
    get_balanced_accuracy_fe(d,y)

    #Feature Selection
##    k = int(len(tfidf_vec.get_feature_names()) / 2)
##    kbest = SelectKBest(chi2,k=k)
##    rfe = RFE(base_model,step=0.05)
##    sfm = SelectFromModel(base_model)
##    d['kbest'] = kbest
##    d['rfe'] = rfe
##    d['sfm'] = sfm
##    benchmark('kbest',base_model, tfidf_train, y, fs=kbest)
##    benchmark('rfe', base_model, tfidf_train, y, fs=rfe)
##    benchmark('sfm', base_model, tfidf_train, y, fs=sfm)
##    get_auroc_fs(d,tfidf_train,y)
##    get_balanced_accuracy_fs(d,tfidf_train,y)
    #Modeling
    lr = LogisticRegression(C=0.5, tol=0.01)
    bnb = BernoulliNB(alpha = 1.0)
    mnb = MultinomialNB(alpha = 1.0)
    rf = RandomForestClassifier(n_estimators=15)

##    d['lr'] = lr
##    d['bnb'] = bnb
##    d['mnb'] = mnb
##    d['rf'] = rf

##    benchmark('lr',lr,tfidf_train,y)
##    benchmark('bnb',bnb,tfidf_train,y)
##    benchmark('mnb',mnb,tfidf_train,y)
##    benchmark('rf',rf,tfidf_train,y)

##    get_auroc(d,tfidf_train,y)
##    get_balanced_accuracy(d,tfidf_train,y)
    
    #Ensembling
##    x_train, x_val, y_train, y_val = train_test_split(tfidf_train, y, test_size=0.4, random_state = 2)
##    get_probability(x_train, y_train, x_val, model = lr).to_csv('q_pairs/lr-prob.csv',index=False)
##    get_probability(x_train, y_train, x_val, model = bnb).to_csv('q_pairs/bnb-prob.csv',index=False)
##    get_probability(x_train, y_train, x_val, model = mnb).to_csv('q_pairs/mnb-prob.csv',index=False)
##    get_probability(x_train, y_train, x_val, model = rf).to_csv('q_pairs/rf-prob.csv',index=False)
##    y_val.to_csv('q_pairs/true-labels.csv',index=False)

##    lr_prob = pd.read_csv('q_pairs/lr-prob.csv')
##    bnb_prob = pd.read_csv('q_pairs/bnb-prob.csv')
##    mnb_prob = pd.read_csv('q_pairs/mnb-prob.csv')
##    rf_prob = pd.read_csv('q_pairs/rf-prob.csv')
##
##    LABELS = ['is_duplicate']
##    ens_prob = lr_prob.copy()
##    ens_prob[LABELS] = (lr_prob[LABELS] + bnb_prob[LABELS]) / 2
##    ens_prob.to_csv('q_pairs/e3.csv', index=False)
##
##    e1 = pd.read_csv('q_pairs/e1.csv')
##    e2 = pd.read_csv('q_pairs/e2.csv')
##    e3 = pd.read_csv('q_pairs/e3.csv')
##    e4 = pd.read_csv('q_pairs/e4.csv')
##    true = pd.read_csv('q_pairs/true-labels.csv')
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
