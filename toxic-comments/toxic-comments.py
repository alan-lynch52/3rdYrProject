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
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

from scipy.sparse import csr_matrix
import string
from unidecode import unidecode
import re
import json

def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    ##import warnings
    ##warnings.filterwarnings("ignore")
    y = train.iloc[0:,:]
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
    k = int(len(vectorizer.get_feature_names())*1)
    print(type(tfidf))
    for label in LABELS:
        tfidf = SelectKBest(chi2, k=k).fit_transform(tfidf,y[label])
    print(type(tfidf))
    #SPLIT INTO TRAINING AND VALIDATION SET
    print("Train test split")
    x_train, x_val, y_train, y_val = train_test_split(tfidf,y, test_size = 0.1, random_state = 2)

    #CREATE MODELS
    print("Init models")
    logreg_model = LogisticRegression()
    nb_model = MultinomialNB()
    #dense_x_train = x_train.todense('C')
    #dense_x_val = x_val.todense('C')
    for label in LABELS:
        logreg_model.fit(x_train, y_train[label])
        nb_model.fit(x_train, y_train[label])
    logreg_scores = []
    nb_scores = []
    for label in LABELS:
        logreg_scores.append(logreg_model.score(x_val,y_val[label]))
        nb_scores.append(nb_model.score(x_val, y_val[label]))
    print(np.mean(logreg_scores))
    print(np.mean(nb_scores))
##    model = buildModel()
##    model.fit(x,y,batch_size=32, epochs = 10, verbose = 1, validation_split = 0.2)
def buildModel():
    cnn = Sequential()
    cnn.add(Embedding(1000, 20))
    cnn.add(Dropout(0.2))
    cnn.add(Conv1D(64, 3, padding='valid',activation='relu'),stride=1)
    cnn.add(GlobalMaxPooling1D())
    cnn.add(Dense(256))
    cnn.add(Dropout(0.2))
    cnn.add(Activation('relu'))
    cnn.add(Dense(1))
    cnn.add(Activation('sigmoid'))
    return cnn

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
