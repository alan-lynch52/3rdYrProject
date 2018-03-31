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
    d['BNB'] = bnb
    d['BNB2'] = bnb
    label = 'toxic'
    get_auroc(d, train_tfidf, y)
    #MAKING PREDICTION
    
    #WRITE TO CSV

    
def make_prediction(x, y, x_test, test_ids,fs=None,model=LogisticRegression()):
    print(x_test.shape)
    print(x.shape)
    pred_dict = collections.OrderedDict()
    print(type(test_ids))
    #pred_dict['id'] = test_ids
    for label in y:
        if fs != None:
            if hasattr(fs, 'transform'):
                x = fs.fit_transform(x,y[label])
                x_test = fs.transform(x_test)
            else:
                fs.fit(x,y[label])
                sfm = SelectFromModel(fs, prefit = True)
                x = sfm.transform(x)
                x_test = sfm.transform(x_test)
        model.fit(x, y[label])
        pred = model.predict_proba(x_test)
        print(pred)
        pred = pred.tolist()
        pred_dict[label] = []
        for item in pred:
            pred_dict[label].append(item[1])
    #print("IDs: {0}".format(pred_dict['id'][1:10]))
##    print("Toxic: {0}".format(pred_dict['toxic'][1:10]))
##    print("Severe Toxic: {0}".format(pred_dict['severe_toxic'][1:10]))
##    print("Threat: {0}".format(pred_dict['threat'][1:10]))
##    print("Insult: {0}".format(pred_dict['insult'][1:10]))
##    print("Identity Hate: {0}".format(pred_dict['identity_hate'][1:10]))
##    print("Dictionary Length: {0}".format(len(pred_dict)))
##    print("Test ids type: {0}".format(type(test_ids)))
    df = pd.DataFrame.from_dict(data = pred_dict)
    df = pd.concat([test_ids, df], axis=1)
    return df 
def benchmark(benchmark_name,model, x, y, fs=None):
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
    accuracy = score_cv_model(model, x, y, cv = None)
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
def get_auroc(models, x, y):
    #split into train and validation set
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.4, random_state=2)
    #train each label
    #predict each label
    d = {}
    index = 1
    size = len(list(y))
    print("size {0}".format(size))
    col = size/2
    row = size/3
    print(row)
    print(col)
    legend_names = list(models.keys())
    print(legend_names)
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
        #plt.annotate(score, xy=(0.5,0.5))
        plt.legend(loc='lower right', prop = {'size':15})
        index += 1
    plt.tight_layout()
    plt.show()
def get_auroc_fe(x_list, y):
    
    index = 1
    size = len(list(y))
    col = size/2
    row = size/3
    print(row)
    print(col)
    legend_names = list(x_list.keys())
    print(legend_names)
    
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
            plt.legend(loc='lower right', prop = {'size':15})
        index += 1
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
