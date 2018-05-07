#https://www.kaggle.com/c/titanic
import sys
sys.path.insert(0, '..')
from toxic_comments import *

def main():
    train = pd.read_csv('titanic_train.csv')
    y = train['Survived']
    y = y.to_frame(name='Survived')
    x = train.drop(['Survived','Name','Cabin','Ticket'],axis=1)
    x = pd.get_dummies(x)
    x = x.fillna(0)
    base_model = LogisticRegression()
    d = collections.OrderedDict()

    #FEATURE SELECTION
##    k = int(x.shape[1] / 2)
##    kbest = SelectKBest(chi2,k=k)
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
##    rf = RandomForestClassifier(n_estimators=15)
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
