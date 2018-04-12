import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC

from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse import hstack
import string
import time
import re
import json
import csv
import collections
import itertools
import sys
sys.path.insert(0, '..')
from toxic_comments import *
def main():
    global CLASSES
    CLASSES = {
        '0':'negative',
        '1':'somewhat negative',
        '2':'neutral',
        '3':'somewhat positive',
        '4':'positive',
        }
    train = pd.read_csv("train.tsv", sep = "\t")
    x = train['Phrase']
    y = train['Sentiment']
    y = pd.get_dummies(y)
    #y = y.to_frame(name="Sentiment")
    tfidf_vec = TfidfVectorizer()
    tfidf_train = tfidf_vec.fit_transform(x)

##    tfidf_char_vec = TfidfVectorizer(analyzer='char')
##    tfidf_char_train = tfidf_char_vec.fit_transform(x)
##    
##    count_vec = CountVectorizer()
##    count_train = count_vec.fit_transform(x)
##
##    bin_vec = TfidfVectorizer(use_idf = False,norm = None, binary = True)
##    bin_train = bin_vec.fit_transform(x)

    base_model = LogisticRegression()

    d = collections.OrderedDict()
    #STACKING
##    tfidf_count_bin = hstack([tfidf_train, count_train, bin_train])
##    tfidf_count = hstack([tfidf_train, count_train])
##    tfidf_bin = hstack([tfidf_train, bin_train])
##    tfidf_word_char = hstack([tfidf_train, tfidf_char_train])
##    d['TFIDF-count-bin'] = tfidf_count_bin
##    d['TFIDF-count'] = tfidf_count
##    d['TFIDF-bin'] = tfidf_bin
##    d['TFIDF-word-char'] = tfidf_word_char
##
##    benchmark('tfidf-count-bin',base_model,tfidf_count_bin,y,fs=None)
##    benchmark('tfidf-count',base_model,tfidf_count,y,fs=None)
##    benchmark('tfidf-bin',base_model,tfidf_bin,y,fs=None)
##    benchmark('tfidf-word-char',base_model,tfidf_word_char,y,fs=None)
##
##    get_balanced_accuracy_fe(d,y)
##    get_auroc_fe(d,y)
    #plot_cm_fe(d,y)
    #FEATURE SELECTION
##    k = int(len(tfidf_vec.get_feature_names()) / 2)
##    print(k)
##    kbest = SelectKBest(chi2, k=k)
##    rfe = RFE(base_model, step = 0.05)
##    sfm = SelectFromModel(base_model)
##    benchmark('kbest',base_model, tfidf_train, y, fs=kbest)
##    benchmark('rfe',base_model, tfidf_train, y, fs=rfe)
##    benchmark('sfm',base_model, tfidf_train, y, fs=sfm)
##
##    d['kbest'] = kbest
##    d['rfe'] = rfe
##    d['sfm'] = sfm
##    
##    get_balanced_accuracy_fs(d,tfidf_train, y)
##    get_auroc_fs(d,tfidf_train, y)
    #MODELING
##    lr = LogisticRegression(C=0.5, tol=0.01)
##    bnb = BernoulliNB(alpha = 1.0)
##    mnb = MultinomialNB(alpha = 1.0)
##    rf = RandomForestClassifier(n_estimators = 15)
##    d['lr'] = lr
##    d['bnb'] = bnb
##    d['mnb'] = mnb
##    d['rf'] = rf

##    benchmark('lr', lr, tfidf_train, y)
##    benchmark('bnb', bnb, tfidf_train, y)
##    benchmark('mnb', mnb, tfidf_train, y)
##    benchmark('rf', rf, tfidf_train, y)
##    get_balanced_accuracy(d, tfidf_train, y)
##    get_auroc(d, tfidf_train, y)

##    x_train, x_val, y_train, y_val = train_test_split(tfidf_train, y, test_size=0.4, random_state = 2)
##    get_probability(x_train, y_train, x_val, model = lr).to_csv('lr-prob.csv',index=False)
##    get_prediction(x_train, y_train, x_val, model = lr).to_csv('lr-pred.csv',index=False)
##
##    get_probability(x_train, y_train, x_val, model = bnb).to_csv('bnb-prob.csv',index=False)
##    get_prediction(x_train, y_train, x_val, model = bnb).to_csv('bnb-pred.csv',index=False)
##
##    get_probability(x_train, y_train, x_val, model = mnb).to_csv('mnb-prob.csv',index=False)
##    get_prediction(x_train, y_train, x_val, model = mnb).to_csv('mnb-pred.csv',index=False)
##
##    get_probability(x_train, y_train, x_val, model = rf).to_csv('rf-prob.csv',index=False)
##    get_prediction(x_train, y_train, x_val, model = rf).to_csv('rf-pred.csv',index=False)
##
##    y_val.to_csv('true-labels',index=False)

##    lr_prob = pd.read_csv('mov_rev/lr-prob.csv')
##    bnb_prob = pd.read_csv('mov_rev/bnb-prob.csv')
##    mnb_prob = pd.read_csv('mov_rev/mnb-prob.csv')
##    rf_prob = pd.read_csv('mov_rev/rf-prob.csv')

##    LABELS = ['0','1','2','3','4']
##
##    ens_prob = lr_prob.copy()
##    ens_prob[LABELS] = (lr_prob[LABELS]*0.7 + bnb_prob[LABELS]*0.1 + mnb_prob[LABELS]*0.1 + rf_prob[LABELS]*0.1)
##    ens_prob.to_csv('e4.csv', index=False)

    e1 = pd.read_csv('mov_rev/e1.csv')
    e2 = pd.read_csv('mov_rev/e2.csv')
    e3 = pd.read_csv('mov_rev/e3.csv')
    e4 = pd.read_csv('mov_rev/e4.csv')
    true = pd.read_csv('mov_rev/true-labels.csv')
##    e1 = e1.round(0)
##    e2 = e2.round(0)
##    e3 = e3.round(0)
##    e4 = e4.round(0)

    d['e1'] = e1
    d['e2'] = e2
    d['e3'] = e3
    d['e4'] = e4
##    get_balanced_accuracy_ensemble(d,true)
    get_auroc_ensemble(d, true)
    
if __name__ == '__main__':
    main()
