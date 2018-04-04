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
sys.path.insert(0, 'U:/Documents/3rdYrProject/3rdYrProject/toxic-comments')
from toxic_comments import *
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
x = tfidf_vec.fit_transform(x)

d = collections.OrderedDict()
d['LR'] = LogisticRegression()
d['MNB'] = MultinomialNB()
d['BNB'] = BernoulliNB()
#get_auroc(d,x,y)
plot_cm(d,x,y)
