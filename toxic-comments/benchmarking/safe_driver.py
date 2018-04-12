import sys
sys.path.insert(0, '..')

from toxic_comments import *

def main():
    train = pd.read_csv('safe_driver_train.csv')
    y = train['target']
    y = y.to_frame(name="target")
    x = train.drop('target', axis = 1)
    del train

    lr = LogisticRegression()
    svc = LinearSVC()
    rf = RandomForestClassifier(n_estimators = 10)
    et = ExtraTreesClassifier()
    gnb = GaussianNB()
    lda = QuadraticDiscriminantAnalysis()
    #benchmark('',lr, x, y)
    #benchmark('',rf, x, y)
    #benchmark('',gnb, x, y)
    benchmark('',rf, x, y)
    #benchmark('', svc, x, y)
if __name__ == '__main__':
    main()
