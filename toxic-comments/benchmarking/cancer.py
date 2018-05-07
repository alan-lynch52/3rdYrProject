import sys
sys.path.insert(0, '..')
from toxic_comments import *
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

def main():
    data = load_breast_cancer()
    x = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=['target'])
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    base_model = LogisticRegression()
    d = collections.OrderedDict()
    print(x.shape)
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
##    d['lr'] = lr
##    d['bnb'] = bnb
##    d['mnb'] = mnb
##    benchmark('lr',lr,x,y)
##    benchmark('bnb',bnb,x,y)
##    benchmark('mnb',mnb,x,y)
##    get_auroc(d,x,y)
##    get_balanced_accuracy(d,x,y)

    #Ensembles
    et = ExtraTreesClassifier(n_estimators=10)
    bag_lr = BaggingClassifier(base_estimator=LogisticRegression(solver='sag',C=0.5, tol=0.01), n_estimators=25)
    rf = RandomForestClassifier(n_estimators=15)
    gb = GradientBoostingClassifier(n_estimators=10)
    d = collections.OrderedDict()
    d['ET'] = et
    d['Bagging'] = bag_lr
    d['RF'] = rf
    d['GB'] = gb
    benchmark('ET',et,x,y)
    benchmark('Bagging',bag_lr,x,y)
    benchmark('RF',rf,x,y)
    benchmark('GB',gb,x,y)
    get_auroc(d,x,y)
    get_balanced_accuracy(d,x, y)
    
if __name__ == '__main__':
    main()
