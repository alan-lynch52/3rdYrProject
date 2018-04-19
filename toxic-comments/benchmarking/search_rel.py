#https://www.kaggle.com/c/crowdflower-search-relevance
import sys
sys.path.insert(0,'..')
from toxic_comments import *
def main():
    train = pd.read_csv('search_rel_train.csv')
    train = train.fillna('')

    query = train['query']
    p_title = train['product_title']
    p_desc = train['product_description']
    y = train['median_relevance'].to_frame('rel')
    y.rel = y.rel.astype(str)
    y = pd.get_dummies(y)
    
    tfidf_vec = TfidfVectorizer()
    count_vec = CountVectorizer()
    bin_vec = TfidfVectorizer(use_idf=False, norm=None, binary=True)
    char_vec = TfidfVectorizer(analyzer='char')

    tfidf_query = tfidf_vec.fit_transform(query)
    tfidf_p_title = tfidf_vec.fit_transform(p_title)
    tfidf_p_desc = tfidf_vec.fit_transform(p_desc)

##    count_query = count_vec.fit_transform(query)
##    count_p_title = count_vec.fit_transform(p_title)
##    count_p_desc = count_vec.fit_transform(p_desc)
##
##    bin_query = bin_vec.fit_transform(query)
##    bin_p_title = bin_vec.fit_transform(p_title)
##    bin_p_desc = bin_vec.fit_transform(p_desc)
##
##    char_query = char_vec.fit_transform(query)
##    char_p_title = char_vec.fit_transform(p_title)
##    char_p_desc = char_vec.fit_transform(p_desc)

    tfidf_train = hstack([tfidf_query, tfidf_p_title, tfidf_p_desc])
##    count_train = hstack([count_query, count_p_title, count_p_desc])
##    bin_train = hstack([bin_query, bin_p_title, bin_p_desc])
##    char_train = hstack([char_query, char_p_title, char_p_desc])
    
    base_model = LogisticRegression()
    d = collections.OrderedDict()
    #Feature Extraction
##    d['TFIDF'] = tfidf_train
##    d['Count'] = count_train
##    d['Binary'] = bin_train
##
##    benchmark('TFIDF',base_model, tfidf_train, y)
##    benchmark('Count',base_model, count_train, y)
##    benchmark('Binary',base_model, bin_train,  y)
##
##    get_auroc_fe(d,y)
##    get_balanced_accuracy_fe(d,y)
    
    #Feature Stacking
##    d['TFIDF-count-bin'] = hstack([tfidf_train,count_train,bin_train])
##    d['TFIDF-count'] = hstack([tfidf_train, count_train])
##    d['TFIDF-bin'] = hstack([tfidf_train, bin_train])
##    d['TFIDF-word-char'] = hstack([tfidf_train, char_train])
##
##    benchmark('TFIDF-count-bin', base_model, d['TFIDF-count-bin'],y)
##    benchmark('TFIDF-count', base_model, d['TFIDF-count'], y)
##    benchmark('TFIDF-bin', base_model, d['TFIDF-bin'], y)
##    benchmark('TFIDF-word-char', base_model, d['TFIDF-word-char'],y)
##    get_auroc_fe(d,y)
##    get_balanced_accuracy_fe(d,y)
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
##
##    benchmark('lr',lr,tfidf_train,y)
##    benchmark('bnb',bnb,tfidf_train,y)
##    benchmark('mnb',mnb,tfidf_train,y)
##    benchmark('rf',rf,tfidf_train,y)
##
##    get_auroc(d,tfidf_train,y)
##    get_balanced_accuracy(d,tfidf_train,y)
    #Ensembling
##    x_train, x_val, y_train, y_val = train_test_split(tfidf_train, y, test_size=0.4, random_state = 2)
##    get_probability(x_train, y_train, x_val, model = lr).to_csv('search_rel/lr-prob.csv',index=False)
##    get_probability(x_train, y_train, x_val, model = bnb).to_csv('search_rel/bnb-prob.csv',index=False)
##    get_probability(x_train, y_train, x_val, model = mnb).to_csv('search_rel/mnb-prob.csv',index=False)
##    get_probability(x_train, y_train, x_val, model = rf).to_csv('search_rel/rf-prob.csv',index=False)
##    y_val.to_csv('search_rel/true-labels.csv',index=False)

    lr_prob = pd.read_csv('search_rel/lr-prob.csv')
    bnb_prob = pd.read_csv('search_rel/bnb-prob.csv')
    mnb_prob = pd.read_csv('search_rel/mnb-prob.csv')
    rf_prob = pd.read_csv('search_rel/rf-prob.csv')

##    LABELS = ['rel_1','rel_2','rel_3','rel_4']
##    ens_prob = lr_prob.copy()
##    ens_prob[LABELS] = (lr_prob[LABELS] + bnb_prob[LABELS]) / 2
##    ens_prob.to_csv('search_rel/e3.csv', index=False)

    e1 = pd.read_csv('search_rel/e1.csv')
    e2 = pd.read_csv('search_rel/e2.csv')
    e3 = pd.read_csv('search_rel/e3.csv')
    e4 = pd.read_csv('search_rel/e4.csv')
    true = pd.read_csv('search_rel/true-labels.csv')
    
    d['e1'] = e1
    d['e2'] = e2
    d['e3'] = e3
    d['e4'] = e4
    get_auroc_ensemble(d, true)
    get_balanced_accuracy_ensemble(d,true)
if __name__ == '__main__':
    main()
