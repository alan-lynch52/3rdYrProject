import sys
sys.path.insert(0, '..')
from toxic_comments import *
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
def main():
    #load training data
    data = load_wine()
    train = pd.DataFrame(np.column_stack((data.data, data.target)), columns= data.feature_names +['target'])
    #split train into x and y
    y = train['target']
    print(train.shape)
    y = y.to_frame('target')
    y['target'] = y['target'].astype(int)
    x = train.drop('target',axis=1)
    clean = {'target': {0: 'wine0', 1 : 'wine1', 2 : 'wine2'}}
    y.replace(clean, inplace=True)
    y = pd.get_dummies(y)

    #standardize data
    scaler = StandardScaler()
    x = scaler.fit_transform(train)
    base_model = LogisticRegression()
    d = collections.OrderedDict()

    #Feature Selection
##    k = int(x.shape[1] / 2)
##    kbest = SelectKBest(mutual_info_classif,k=k)
##    rfe = RFE(base_model,step=1)
##    sfm = SelectFromModel(base_model)
##    d['kbest'] = kbest
##    d['rfe'] = rfe
##    d['sfm'] = sfm
##    benchmark('kbest',base_model, x, y, fs=kbest)
##    benchmark('rfe', base_model, x, y, fs=rfe)
##    benchmark('sfm', base_model, x, y, fs=sfm)
##    get_auroc_fs(d,x,y)
##    get_balanced_accuracy_fs(d,x,y)
    
    #Modeling
##    lr = LogisticRegression(C=0.5, tol=0.01)
##    bnb = BernoulliNB(alpha = 1.0)
##    mnb = MultinomialNB(alpha = 1.0)
##    svc = SVC(probability=True)
##    d['lr'] = lr
##    d['bnb'] = bnb
##    d['mnb'] = mnb
##    benchmark('lr',lr,x,y)
##    benchmark('bnb',bnb,x,y)
##    benchmark('mnb',mnb,x,y)
##    get_auroc(d, x, y)
##    get_balanced_accuracy(d, x, y)

    #Ensembles
##    et = ExtraTreesClassifier(n_estimators=10)
##    bag_lr = BaggingClassifier(base_estimator=LogisticRegression(solver='sag',C=0.5, tol=0.01), n_estimators=25)
##    rf = RandomForestClassifier(n_estimators=15)
##    gb = GradientBoostingClassifier(n_estimators=10)
##    d = collections.OrderedDict()
##    d['ET'] = et
##    d['Bagging'] = bag_lr
##    d['RF'] = rf
##    d['GB'] = gb
##    benchmark('ET',et,x,y)
##    benchmark('Bagging',bag_lr,x,y)
##    benchmark('RF',rf,x,y)
##    benchmark('GB',gb,x,y)
##    get_auroc(d,x,y)
##    get_balanced_accuracy(d,x, y)
if __name__ == '__main__':
    main()
