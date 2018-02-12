import pandas as pd
import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

from boruta import BorutaPy

from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
import string
from unidecode import unidecode
import re
import json

def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    ##import warnings
    ##warnings.filterwarnings("ignore")
    y = train.iloc[:,2:]
    print(list(y))
    LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    x = train.drop(labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate'],axis=1)
    x['comment_text'].fillna("unknown", inplace=True)
    x = x.drop(labels = ['id'],axis = 1)
    
    del train

    #FEATURE EXTRACTION
    
    x = strip_text(x)
    #get bag of words
    #bag_of_words(clean_x)
    
    word_bag = read_dict("bag_of_words.json")
    #term_frequencies(word_bag, clean_x)
    #idf(word_bag, clean_x)

    #GET TFIDF scores
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(x)
    #FEATURE SELECTION
    print("SelectKBest")
    k = int(len(vectorizer.get_feature_names())*0.5)
    for label in LABELS:
        tfidf_kbest = SelectKBest(chi2, k=k).fit_transform(tfidf,y[label])
    print("Boruta")
    rf = RandomForestClassifier()
    save_npz('tfidf.npz', tfidf)
    #write_sparse('tfidf.txt', tfidf)
    #tfidf_dense = tfidf.todense()
##    for label in LABELS:
##        tfidf_boruta = BorutaPy(rf, n_estimators = 'auto',
##                                random_state = 2).fit(tfidf, y[label])
##    tfidf_boruta.transform(tfidf)
##    print("RFE")
##    for label in LABELS:
##        print("RFE on label: %s" % label)
##        tfidf_rfe  = RFE(MultinomialNB(), k, step=1).fit_transform(tfidf,y[label])
    
    #SPLIT INTO TRAINING AND VALIDATION SET
    print("Train test split")
    x_train, x_val, y_train, y_val = train_test_split(tfidf_kbest,y, test_size = 0.1, random_state = 2)

    #CREATE MODELS
    print("Init models")
    logreg_model = build_model(LogisticRegression(), x_train, y_train)
    logreg_scores = score_model(logreg_model, x_val, y_val)
    nb_model = build_model(MultinomialNB(), x_train, y_train)
    nb_scores = score_model(nb_model, x_val, y_val)
    print(logreg_scores)
    print(nb_scores)

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
