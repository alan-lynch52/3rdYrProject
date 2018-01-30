import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.fftpack import dct

import warnings
warnings.filterwarnings("ignore")

def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    y = train["label"]
    x = train.drop(labels = ["label"], axis = 1)

    del train

    print(y)

    print(x)
    random_seed = 2
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = random_seed)

    n_estimators_array = np.array([500,600,700,800,900,1000,1200])
    n_samples = 10
    n_grid = len(n_estimators_array)

    score_array_mu = np.zeros(n_grid)
    score_array_sigma = np.zeros(n_grid)

    j=0
    for n_estimators in n_estimators_array:
        score_array = np.zeros(n_samples)
        for i in range(0, n_samples):
            clf = RandomForestClassifier(n_estimators = n_estimators, n_jobs=1, criterion="gini")
            score_array[i] = evaluate_classifier(clf, x.iloc[0:1000],y.iloc[0:1000],0.8)
        score_array_mu[j], score_array_sigma[j] = np.mean(score_array), np.std(score_array)
        j=j+1


    plt.figure(figsize =(7,3))
    plt.errorbar(n_estimators_array, score_array_mu, yerr=score_array_sigma, fmt='k.-')
    plt.xscale("log")
    plt.xlabel("number of estimators", size = 20)
    plt.ylabel("accuracy", size=20)
    plt.xlim(400, 1300)
    plt.grid(which="both")
    plt.show()
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    print(rf.predict(x_test))

def evaluate_classifier(clf, data, target, split_ratio):
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=split_ratio, random_state=0)
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test)

    y = train["label"]
    x = train.drop(labels = ["label"], axis = 1)
    
    x = x.iloc[:,:]
    #disp_img(y.iloc[1],x.iloc[1])

    x = dct(x)
    del train

    random_seed = 2
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size = 0.1, random_state = random_seed)

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)


    print(rf.score(x_validation, y_validation))
    #random forest base model: 93.97% accuracy
    


def disp_img(lbl, px):
    img = px.as_matrix()
    img = img.reshape((28,28))
    plt.imshow(img, cmap='gray')
    plt.title(lbl)
    plt.show()

if __name__ == "__main__":
    main()
