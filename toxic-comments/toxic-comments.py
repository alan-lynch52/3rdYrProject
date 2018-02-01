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
    word_bag = {}
    incrementer = 0
    remove = ",/+-_="
    replace = "      "
##    patterns = re.compile("["
##                u"\U0001F600-\U0001F64F"
##                u"\U0001F300-\U0001F5FF"
##                u"\U0001F680-\U0001F6FF"
##                u"\U0001F1E0-\U0001F1FF"
##                "]+", flags=re.UNICODE)
    emoji_strip = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    transTable = str.maketrans(remove, replace)
    for comment in x['comment_text']:
        comment = comment.translate(transTable)
        comment = emoji_strip.sub(r'',comment)
        #print(comment)
        for word in comment.split():
            if word not in word_bag:
                word_bag[word] = incrementer
                incrementer += 1
    print(word_bag)
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







if __name__ == "__main__":
    main()
