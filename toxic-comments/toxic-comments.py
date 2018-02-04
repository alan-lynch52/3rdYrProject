import pandas as pd
import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils

import string
from unidecode import unidecode
import re
import json

def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    ##import warnings
    ##warnings.filterwarnings("ignore")
    y = train.iloc[1:,:]
    x = train.drop(labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate'],axis=1)
    x = x.drop(labels = ['id'],axis = 1)

    del train

    #print(y)
    #print(x)

    #FEATURE EXTRACTION
    #get bag of words
    clean_x = strip_text(x)
    #print(clean_x)
    word_bag = bag_of_words(clean_x)
    #print(word_bag)

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

def bw_writer(d):
    with open('bag_of_words.txt','w') as outfile:
        try:
            json.dump(d, outfile)
        except TypeError:
            print("Type Error found")
            

def bw_reader(filepath):
    with open(filepath,'r') as file:
        return file
def strip_text(x):
    commentList = []
    remove = ",/\+-_='.:()\"[]{}!"
    replace = "                  "
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
    bw_writer(word_bag)
    return word_bag
def term_freq(term,comment):
    for word in comment.split():
        print(word)
if __name__ == "__main__":
    main()
