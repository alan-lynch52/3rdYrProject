#https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
import sys
sys.path.insert(0, '..')

from toxic_comments import *

def main():
    train = pd.read_csv('safe_driver_train.csv')
    y = train['target']
    y = y.to_frame(name="target")
    x = train.drop('target', axis = 1)
    print(x.head())
    base_model = LogisticRegression()
    #Feature Selection
##    k = int(x.shape[1]/2)
##    print(k)
##    kbest = SelectKBest(mutual_info_classif,k=k)
##    rfe = RFE(base_model, step=1)
##    sfm = SelectFromModel(base_model)
##    d = collections.OrderedDict()
##    d['kbest'] = kbest
##    d['sfm'] = rfe
##    d['sfm'] = sfm
##    for label in y:
##        print(label)
##    benchmark('kbest', base_model, x, y, fs=kbest)
##    benchmark('rfe', base_model, x, y, fs=rfe)
##    benchmark('sfm', base_model, x, y, fs=sfm)
##    get_balanced_accuracy_fs(d,x,y)
##    get_auroc_fs(d,x,y)
    

    #modelling
    lr = LogisticRegression()
    mnb = MultinomialNB()
    bnb = BernoulliNB()
    d = collections.OrderedDict()
    d['lr'] = lr
    d['bnb'] = bnb
    d['mnb'] = mnb
    benchmark('lr',lr, x, y)
    benchmark('bnb',bnb,x,y)
    benchmark('mnb', mnb, x, y)

    get_auroc(d,x,y)
    get_balanced_accuracy(d,x,y)
if __name__ == '__main__':
    main()
