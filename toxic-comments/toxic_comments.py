import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import *
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
def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    ##import warnings
    ##warnings.filterwarnings("ignore")
    y = train.iloc[:,2:]
    LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    x = train.drop(labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate'],axis=1)
    x['comment_text'].fillna("unknown", inplace=True)
    x = x.drop(labels = ['id'],axis = 1)
    test_ids = test.drop(labels = ['comment_text'], axis = 1)
    test = test.drop(labels = ['id'],axis= 1)
    test['comment_text'].fillna("unknown", inplace=True)
    del train
    
    df = pd.concat([x['comment_text'], test['comment_text']], axis=0)
    #FEATURE EXTRACTION EXPERIMENTS

    #GET BINARY FEATURE
    binary_vec = TfidfVectorizer(use_idf = False,norm = None, binary = True)
    binary_vec.fit(df)
    bin_train = binary_vec.transform(x['comment_text'])
    bin_test = binary_vec.transform(test['comment_text'])

    #GET TFIDF FEATURE
    tfidf_vec = TfidfVectorizer()
    tfidf_vec.fit(df)
    train_tfidf = tfidf_vec.transform(x['comment_text'])
    test_tfidf = tfidf_vec.transform(test['comment_text'])
    #Get benchmarks
    model = LogisticRegression()
    mnb = MultinomialNB()
    bnb = BernoulliNB()
    d = collections.OrderedDict()
    d['LR'] = model
    d['MNB'] = mnb
##    d['BNB'] = bnb
##    d['BNB2'] = bnb
    #get_auroc(d, train_tfidf, y)
    
    #plot_cm(d, train_tfidf, y)
    #MAKING PREDICTION
    x_train, x_val, y_train, y_val = train_test_split(train_tfidf, y, test_size = 0.4, random_state = 2)
    preds = get_prediction(x_train, y_train, x_val, model=model)
    probs = get_probability(x_train, y_train, x_val, model=model)
    preds.to_csv('val/LR_Preds.csv', index=False)
    probs.to_csv('val/LR_Probs.csv',index=False)
    y_val.to_csv('val/LR_True.csv', index=False)
    
    #WRITE TO CSV

    
def get_probability(x, y, x_test, test_ids=None,fs=None,model=LogisticRegression()):
    print(x_test.shape)
    print(x.shape)
    pred_dict = collections.OrderedDict()
    print(type(test_ids))
    for label in y:
        if fs != None:
            if hasattr(fs, 'transform'):
                new_x = fs.fit_transform(x,y[label])
                new_x_test = fs.transform(x_test)
            else:
                fs.fit(x,y[label])
                sfm = SelectFromModel(fs, prefit = True)
                new_x = sfm.transform(x)
                new_x_test = sfm.transform(x_test)
        else:
            new_x = x
            new_x_test = x_test
        model.fit(new_x, y[label])
        pred = model.predict_proba(new_x_test)[:,1]
        pred_dict[label] = pred
    df = pd.DataFrame.from_dict(data = pred_dict)
    if test_ids is not None:
        df = pd.concat([test_ids, df], axis=1)
    return df
def get_prediction(x, y, x_test, test_ids=None,fs=None,model=LogisticRegression()):
    print(x_test.shape)
    print(x.shape)
    pred_dict = collections.OrderedDict()
    print(type(test_ids))
    for label in y:
        if fs != None:
            if hasattr(fs, 'transform'):
                new_x = fs.fit_transform(x,y[label])
                new_x_test = fs.transform(x_test)
            else:
                fs.fit(x,y[label])
                sfm = SelectFromModel(fs, prefit = True)
                new_x = sfm.transform(x)
                new_x_test = sfm.transform(x_test)
        else:
            new_x = x
            new_x_test = x_test
        model.fit(new_x, y[label])
        pred = model.predict(new_x_test)
        pred_dict[label] = pred
    df = pd.DataFrame.from_dict(data = pred_dict)
    if test_ids is not None:
        df = pd.concat([test_ids, df], axis=1)
    return df
def benchmark(benchmark_name,model, x, y, fs=None,cv=3):
    benchmarks = {}
    start = time.time()
    for label in y:
        if fs != None:
            fs.fit(x,y[label])
            if hasattr(fs, 'transform'):
                x = fs.transform(x)
            else:
                sfm = SelectFromModel(fs, prefit=True)
                x = sfm.transform(x)
    #x_train, x_val, y_train, y_val = train_test_split(new_x, y, test_size = 0.1, random_state= 2)

    #fitted_model = build_model(model, x_train, y_train)
    print(x.shape)
    accuracy = score_cv_model(model, x, y, cv = cv)
    end = time.time()
    duration = end-start
    print("Benchmark Title: {0}".format(benchmark_name))
    print("Accuracy is: {0}".format(accuracy))
    print("Number of attributes is: {0}".format(x.shape[1]))
    print("Time taken: {0}".format(duration))
    benchmarks['name'] = benchmark_name
    benchmarks['accuracy'] = round(accuracy,4)
    benchmarks['num_attributes'] = x.shape[1]
    benchmarks['duration'] = round(duration,2)
    return benchmarks
def plot_cm(models, x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.4, random_state=2)
    index = 1
    size = len(list(models))
    col = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    row = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    print(row)
    print(col)
    for key in models:
        c_preds = np.array([])
        c_true = np.array([])
        for label in y:
            model = models[key]
            model.fit(x_train, y_train[label])
            preds = model.predict(x_val)
            true = y_val[label]
            c_preds = np.append(c_preds,preds)
            c_true = np.append(c_true,true)
        cm = confusion_matrix(c_true,c_preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        plt.subplot(row, col, index)
        plt.imshow(cm,interpolation="nearest",cmap=plt.cm.Blues)
        plt.colorbar()
        plt.title(key)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        negative = "0"
        positive = "1"
        conds = [negative,positive]
        ticks = np.arange(len(conds))
        plt.xticks(ticks,conds)
        plt.yticks(ticks,conds)
        thresh = cm.max() / 2
        fmt = '.2f'
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color = "white" if cm[i,j] > thresh else "black")
        index += 1
    plt.tight_layout()
    plt.show()
def plot_cm_fs(fs_list, x, y):
    model = LogisticRegression()
    index = 1
    size = len(list(fs_list))
    col = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    row = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    for key in fs_list:
        c_preds = np.array([])
        c_true = np.array([])
        for label in y:
            fs = fs_list[key]
            fs.fit(x, y[label])
            new_x = fs.transform(x)
            x_train, x_val, y_train, y_val = train_test_split(new_x, y, test_size=0.4, random_state=2)
            model.fit(x_train, y_train[label])
            preds = model.predict(x_val)
            true = y_val[label]
            c_preds = np.append(c_preds,preds)
            c_true = np.append(c_true,true)
        cm = confusion_matrix(c_true,c_preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        plt.subplot(row, col, index)
        plt.imshow(cm,interpolation="nearest",cmap=plt.cm.Blues)
        plt.colorbar()
        plt.title(key)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        negative = "0"
        positive = "1"
        conds = [negative,positive]
        ticks = np.arange(len(conds))
        plt.xticks(ticks,conds)
        plt.yticks(ticks,conds)
        thresh = cm.max() / 2
        fmt = '.2f'
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color = "white" if cm[i,j] > thresh else "black")
        index += 1
    plt.tight_layout()
    plt.show()
def plot_cm_ensemble(ensemble_pred_list, true_list):
    index = 1
    size = len(list(ensemble_pred_list))
    col = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    row = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    for key in ensemble_pred_list:
        ensemble = ensemble_pred_list[key]
        c_preds = np.array([])
        c_true = np.array([])
        for label in true_list:
            preds = ensemble[label]
            true = true_list[label]
            c_preds = np.append(c_preds,preds)
            c_true = np.append(c_true,true)
        cm = confusion_matrix(c_true,c_preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        plt.subplot(row, col, index)
        plt.imshow(cm,interpolation="nearest",cmap=plt.cm.Blues)
        plt.colorbar()
        plt.title(key)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        negative = "0"
        positive = "1"
        conds = [negative,positive]
        ticks = np.arange(len(conds))
        plt.xticks(ticks,conds)
        plt.yticks(ticks,conds)
        thresh = cm.max() / 2
        fmt = '.2f'
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color = "white" if cm[i,j] > thresh else "black")
        index += 1
    plt.tight_layout()
    plt.show()
def plot_cm_fe(x_list, y):
    model = LogisticRegression()
    index = 1
    size = len(list(x_list))
    col = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    row = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    for key in x_list:
        x = x_list[key]
        c_preds = np.array([])
        c_true = np.array([])
        for label in y:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.4, random_state=2)
            model.fit(x_train, y_train[label])
            preds = model.predict(x_val)
            true = y_val[label]
            c_preds = np.append(c_preds,preds)
            c_true = np.append(c_true,true)
        cm = confusion_matrix(c_true,c_preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        plt.subplot(row, col, index)
        plt.imshow(cm,interpolation="nearest",cmap=plt.cm.Blues)
        plt.colorbar()
        plt.title(key)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        negative = "0"
        positive = "1"
        conds = [negative,positive]
        ticks = np.arange(len(conds))
        plt.xticks(ticks,conds)
        plt.yticks(ticks,conds)
        thresh = cm.max() / 2
        fmt = '.2f'
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color = "white" if cm[i,j] > thresh else "black")
        index += 1
    plt.tight_layout()
    plt.show()
def get_balanced_accuracy(models, x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.4, random_state=2)
    c_preds = np.array([])
    c_true = np.array([])
    bacc_list = {}
    for key in models:
        model = models[key]
        for label in y:
            model.fit(x_train, y_train[label])
            preds = model.predict(x_val)
            true = y_val[label]
            c_preds = np.append(c_preds, preds)
            c_true = np.append(c_true, true)
        #roc_auc is equivalent to balanced accuracy given binary preds and true labels
        bacc = roc_auc_score(c_true, c_preds)
        bacc = round(bacc,4)
        bacc_list[key] = bacc
    print(bacc_list)
    return bacc_list
def get_balanced_accuracy_fs(fs_list, x, y):
    model = LogisticRegression()
    c_preds = np.array([])
    c_true = np.array([])
    bacc_list = {}
    for key in fs_list:
        fs = fs_list[key]
        for label in y:
            fs.fit(x,y[label])
            new_x = fs.transform(x)
            x_train, x_val, y_train, y_val = train_test_split(new_x, y, test_size=0.4, random_state=2)
            model.fit(x_train, y_train[label])
            preds = model.predict(x_val)
            true = y_val[label]
            c_preds = np.append(c_preds, preds)
            c_true = np.append(c_true, true)
        #roc_auc is equivalent to balanced accuracy given binary preds and true labels
        bacc = roc_auc_score(c_true, c_preds)
        bacc = round(bacc,4)
        bacc_list[key] = bacc
    print(bacc_list)
    return bacc_list
def get_balanced_accuracy_fe(x_list, y):
    model = LogisticRegression()
    c_preds = np.array([])
    c_true = np.array([])
    bacc_list = {}
    for key in x_list:
        x = x_list[key]
        for label in y:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.4, random_state=2)
            model.fit(x_train, y_train[label])
            preds = model.predict(x_val)
            true = y_val[label]
            c_preds = np.append(c_preds, preds)
            c_true = np.append(c_true, true)
        #roc_auc is equivalent to balanced accuracy given binary preds and true labels
        bacc = roc_auc_score(c_true, c_preds)
        bacc = round(bacc,4)
        bacc_list[key] = bacc
    print(bacc_list)
    return bacc_list
def get_balanced_accuracy_ensemble(ensemble_pred_list, y):
    c_preds = np.array([])
    c_true = np.array([])
    bacc_list = {}
    for key in ensemble_pred_list:
        ensemble = ensemble_pred_list[key]
        for label in y:
            preds = ensemble[label]
            true = y[label]
            c_preds = np.append(c_preds, preds)
            c_true = np.append(c_true, true)
        #roc_auc is equivalent to balanced accuracy given binary preds and true labels
        bacc = roc_auc_score(c_true, c_preds)
        bacc = round(bacc,4)
        bacc_list[key] = bacc
    print(bacc_list)
    return bacc_list
def get_auroc(models, x, y):
    #split into train and validation set
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.4, random_state=2)
    #train each label
    #predict each label
    d = {}
    index = 1
    print(type(y))
    size = len(list(y))
    print("size {0}".format(size))
    col = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    row = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    print(row)
    print(col)
    legend_names = list(models.keys())
    print(legend_names)
    score_dict = {}
    for key in models:
        score_dict[key] = []
    for label in y:
        plt.subplot(row,col,index)
        plt.title(label)
        for key in models:
            model = models[key]
            model.fit(x_train, y_train[label])
            preds = model.predict_proba(x_val)[:,1]
            fpr, tpr, th = roc_curve(y_val[label],preds)
            score = roc_auc_score(y_val[label],preds)
            plt.plot(fpr,tpr, label=key)
            score = round(score,4)
            score_dict[key].append(score)
        #plt.annotate(score, xy=(0.5,0.5))
        plt.legend(loc='lower right', prop = {'size':15})
        index += 1
    for key in score_dict:
        print(key)
        print(np.mean(score_dict[key]))
    plt.tight_layout()
    plt.show()
def get_auroc_fe(x_list, y):
    index = 1
    size = len(list(y))
    col = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    row = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    print(row)
    print(col)
    legend_names = list(x_list.keys())
    print(legend_names)
    score_dict = {}
    for key in x_list:
        score_dict[key] = []
    for label in y:
        plt.subplot(row,col,index)
        plt.title(label)
        for key in x_list:
            model = LogisticRegression()
            x = x_list[key]
            #split into train and validation set
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.4, random_state=2)
            #train each label
            #predict each label
            model.fit(x_train, y_train[label])
            preds = model.predict_proba(x_val)[:,1]
            fpr, tpr, th = roc_curve(y_val[label],preds)
            score = roc_auc_score(y_val[label],preds)
            plt.plot(fpr,tpr, label=key)
            score = round(score,4)
            score_dict[key].append(score)
            plt.legend(loc='lower right', prop = {'size':12})
        index += 1
    for key in score_dict:
        print(key)
        print(np.mean(score_dict[key]))
    plt.tight_layout()
    plt.show()
def get_auroc_fs(fs_list, x, y):
    index = 1
    size = len(list(y))
    col = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    row = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    legend_names = list(fs_list.keys())
    score_dict = {}
    for key in fs_list:
        score_dict[key] = []
    for label in y:
        plt.subplot(row,col,index)
        plt.title(label)
        for key in fs_list:
            print(key)
            model = LogisticRegression()
            fs = fs_list[key]
            fs.fit(x,y[label])
            if hasattr(fs, "transform"):
                new_x = fs.transform(x)
            else:
                sfm = SelectFromModel(fs, prefit=True)
                new_x = sfm.transform(x)
            x_train, x_val, y_train, y_val = train_test_split(new_x, y, test_size = 0.4, random_state=2)
            model.fit(x_train,y_train[label])
            preds = model.predict_proba(x_val)[:,1]
            fpr, tpr, th = roc_curve(y_val[label],preds)
            score = roc_auc_score(y_val[label],preds)
            plt.plot(fpr,tpr, label=key)
            score = round(score, 4)
            score_dict[key].append(score)
            plt.legend(loc="lower right", prop = {'size':15})
        index += 1
    for key in score_dict:
        print(key)
        print(np.mean(score_dict[key]))
    plt.tight_layout()
    plt.show()
def get_auroc_ensemble(ensemble_pred_list, true_list):
    index = 1
    size = len(list(true_list))
    col = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    row = size/2 if size > 1 and size%2 == 0 else int(size/2)+1
    score_dict = {}
    for key in ensemble_pred_list:
        score_dict[key] = []
    for label in true_list:
        plt.subplot(row, col, index)
        plt.title(label)
        c_score = np.array([])
        for key in ensemble_pred_list:
            ensemble = ensemble_pred_list[key]
            preds = ensemble[label]
            true = true_list[label]
            fpr, tpr, th = roc_curve(true,preds)
            score = roc_auc_score(true, preds)
            score_dict[key].append(score)
            plt.plot(fpr, tpr, label=key)
            plt.legend(loc="lower right", prop = {'size':15})
        index += 1
    for key in score_dict:
        print(key)
        print(np.mean(score_dict[key]))
    plt.tight_layout()
    plt.show()
def barplot_benchmark(index, y, title, xticks, ylim=None):
    plt.bar(np.arange(index), y)
    plt.ylim(ylim)
    plt.title(title)
    print(xticks)
    print(type(xticks))
    plt.xticks(np.arange(index), xticks)
    plt.show()
def build_cnn():
    cnn = Sequential()
    #cnn.add(Conv1D(filters = 32, kernel_size = (5,5), padding = 'Same', activation='relu', input_shape = input_shape))
    cnn.add(MaxPool1D(pool_size = (2,2), strides = (2,2)))
    cnn.add(Dropout(0.25))

    cnn.add(Flatten())
    cnn.add(Dense(256, activation = 'relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(10, activation = 'softmax'))

    optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay=0.0)
    cnn.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return cnn
def sgd_best_alpha(alphas, x, y):
    scores = []
    base_model = LogisticRegression()
    for alpha in alphas:
        fs = SGDClassifier(alpha=alpha,penalty='elasticnet')
        for label in y:
            fs.fit(x,y[label])
        sfm = SelectFromModel(fs, prefit=True)
        new_x = sfm.transform(x)
        scores.append(score_cv_model(base_model, new_x, y, cv = 5))
        del fs, sfm, new_x
    plt.plot(alphas,scores)
    plt.show()
def kbest_find_k(kList, x, y):
    scores = []
    base_model = LogisticRegression()
    for k in kList:
            clf = SelectKBest(chi2, k=k)
            for label in y:
                clf.fit(x,y[label])
            new_x = clf.transform(x)
            scores.append(score_cv_model(base_model, new_x, y, cv = 5))
    plt.plot(kList,scores)
    plt.show()
def lr_find_c(cList, x,y):
    scores = []
    base_model = LogisticRegression()
    for c in cList:
        clf = LogisticRegression(penalty = 'l1', C=c)
        for label in y:
            clf.fit(x,y[label])
        sfm = SelectFromModel(clf, prefit=True)
        new_x = sfm.transform(x)
        scores.append(score_cv_model(base_model, new_x, y, cv = 5))
    plt.plot(cList, scores)
    plt.show()
def ridge_find_alpha(alphas, x, y):
    scores = []
    base_model = LogisticRegression()
    for alpha in alphas:
        clf = RidgeClassifier(alpha=alpha)
        for label in y:
            clf.fit(x,y[label])
        sfm = SelectFromModel(clf, prefit=True)
        new_x = sfm.transform(x)
        scores.append(score_cv_model(base_model, new_x, y, cv = 5))
    plt.plot(alphas, scores)
    plt.show()
def build_model(model, x, y):
    print(list(y))
    for label in list(y):
        print(label)
        model.fit(x, y[label])
    return model
def score_model(model, x, y):
    scores = []
    for label in y:
        score = model.score(x, y[label])
        scores.append(score)
    return np.mean(scores)
def score_cv_model(model, x, y, cv=None):
    scores = []
    for label in y:
        score = cross_val_score(model, x, y[label], cv = cv, n_jobs=-1)
        scores.append(np.mean(score))
    return np.mean(scores)
def write_dict(d, filepath):
    with open(filepath,'w') as outfile:
        try:
            json.dump(d, outfile)
        except TypeError:
            print("Type Error found")
def read_dict(filepath):
    with open(filepath,'r') as file:
        data = json.load(file)
        return data
def write_dict_to_csv(d, filepath):
    with open(filepath, 'a') as csvfile:
        fieldnames = d.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,lineterminator='\n')
        writer.writerow(d)
def write_sparse(array, filepath):
    np.savez(filepath, data=array.data, indices = array.indices,
             indptr = array.indptr, shape = array.shape)
def read_sparse(filepath):
    f = np.load(filepath)
    return csr_matrix((f['data'], f['indices'], f['indptr']), shape = f['shape'])
def strip_text(x):
    commentList = []
    remove = ",/\+-_=.:()\"[]{}!"
    replace = "                 "
    emoji_strip = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    transTable = str.maketrans(remove, replace)
    #pattern = re.compile('[\W_]+')
    for comment in x['comment_text']:
        re.sub('\W+', '',comment)
        comment = comment.translate(transTable)
        comment = emoji_strip.sub(r'',comment)
        comment = re.sub(r'\t', ' ', comment)
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'\r', ' ', comment)
        comment = re.sub("'", '', comment)
        comment = comment.lower()
        comment = unidecode(comment)
        comment.replace(r'\"', '')
        #comment = re.split('\W+',comment)
        commentList.append(comment)
    return commentList
def bag_of_words(clean_x):
    word_bag = {}
    incrementer = 0
    for comment in clean_x:
        for word in comment.split():
            if word not in word_bag:
                word_bag[word] = incrementer
                incrementer += 1
    write_dict(word_bag, 'bag_of_words.json')
    return word_bag
#takes in a bag of words dictionary and a list of cleaned comments and produces
# a json file of term freqs for each comment
def term_frequencies(word_bag, comments):
    tfDict = {}
    commentID = 1
    for comment in comments:
        tfDict[commentID] = {}
        #go through each word and count occurences
        for word in comment.split():
            if word not in tfDict[commentID]:
                tfDict[commentID][word] = 1
            else:
                tfDict[commentID][word] += 1
        commentID += 1
    write_dict(tfDict,'term_freqs.json')
#given a bag of words and a corpus, generates a json file containg inverse document frequencies for every term in the bag of words
def idf(word_bag, comments):
    idf_dict = {}
    N = len(comments)
    for term in word_bag:
        nt = term_doc_count(term, comments)
        idf_dict[term] = np.log(N/nt)
    write_dict(idf_dict, "inverse_doc_freqs.json")
    return idf_dict
#gets the count of documents that contain a given term from a given corpus
def term_doc_count(term, comments):
    doc_count = 1
    for comment in comments:
        if term in comment:
            doc_count += 1
    return doc_count
if __name__ == "__main__":
    main()
