from scipy.io.arff import loadarff
import sys
sys.path.insert(0, '..')
from toxic_comments import *

def main():
    data = loadarff('amazon_reviews.arff')
    train = pd.DataFrame(data[0])
    y = train['class']
    y = pd.get_dummies(y)
    x = train.drop(['class'],axis=1)
    base_model = LogisticRegression()
    d= collections.OrderedDict()
    print(train.shape)
    #FEATURE SELECTION
#FEATURE SELECTION
##    k = int(train.shape[1] / 2)
##    print(k)
##    kbest = SelectKBest(chi2, k=k)
##    rfe = RFE(base_model, step = 0.05)
##    sfm = SelectFromModel(base_model)
##    benchmark('kbest',base_model, x, y, fs=kbest)
##    benchmark('rfe',base_model, x, y, fs=rfe)
##    benchmark('sfm',base_model, x, y, fs=sfm)
##
##    d['kbest'] = kbest
##    d['rfe'] = rfe
##    d['sfm'] = sfm
##    
##    get_balanced_accuracy_fs(d,x, y)
##    get_auroc_fs(d,x, y)
    #MODELING
##    lr = LogisticRegression(C=0.5, tol=0.01)
##    bnb = BernoulliNB(alpha = 1.0)
##    mnb = MultinomialNB(alpha = 1.0)
##    d['lr'] = lr
##    d['bnb'] = bnb
##    d['mnb'] = mnb
##
##    benchmark('lr', lr, x, y)
##    benchmark('bnb', bnb, x, y)
##    benchmark('mnb', mnb, x, y)
##    get_balanced_accuracy(d, x, y)
##    get_auroc(d, x, y)

    #ENSEMBLES
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
##    
if __name__ == '__main__':
    main()
