#https://www.kaggle.com/c/crowdflower-search-relevance
import sys
sys.path.insert(0,'..')
from toxic_comments import *
def main():
    #load train data
    train = pd.read_csv('search_rel_train.csv')
    #fill missing instances
    train = train.fillna('')
    #load into query, p_title and p_desc - ready for feature extraction
    query = train['query']
    p_title = train['product_title']
    p_desc = train['product_description']
    #get target, turn into binary classifications
    y = train['median_relevance'].to_frame('rel')
    y.rel = y.rel.astype(str)
    y = pd.get_dummies(y)
    
    tfidf_vec = TfidfVectorizer()
    count_vec = CountVectorizer()
    bin_vec = TfidfVectorizer(use_idf=False, norm=None, binary=True)
    char_vec = TfidfVectorizer(analyzer='char')
    #extract features from query, p_title and p_desc
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
    
    #stack into feature vectors ready for experiments
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
##    benchmark('TFIDF',base_model, tfidf_train, y)
##    benchmark('Count',base_model, count_train, y)
##    benchmark('Binary',base_model, bin_train,  y)
##    get_auroc_fe(d,y)
##    get_balanced_accuracy_fe(d,y)
    #Feature Stacking
##    d['TFIDF-count-bin'] = hstack([tfidf_train,count_train,bin_train])
##    d['TFIDF-count'] = hstack([tfidf_train, count_train])
##    d['TFIDF-bin'] = hstack([tfidf_train, bin_train])
##    d['TFIDF-word-char'] = hstack([tfidf_train, char_train])
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
##    benchmark('lr',lr,tfidf_train,y)
##    benchmark('bnb',bnb,tfidf_train,y)
##    benchmark('mnb',mnb,tfidf_train,y)
##    benchmark('rf',rf,tfidf_train,y)
##    get_auroc(d,tfidf_train,y)
##    get_balanced_accuracy(d,tfidf_train,y)
    #Ensembling
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
