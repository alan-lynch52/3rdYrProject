import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.fftpack import dct
from sklearn.decomposition import pca
def main():
    train = pd.read_csv("train.csv")
    #test = pd.read_csv("test.csv")

    y = train["label"]
    x = train.drop(labels = ["label"], axis = 1)
    
    x = x.iloc[:,:]
    #disp_img(y.iloc[1],x.iloc[1])

    x_dct = dct(x)
    x_pca = pca(n_components = 10)
    x_pca.fit(x)
    
    del train

    random_seed = 2
    x_train, x_validation, y_train, y_validation = train_test_split(x_pca, y, test_size = 0.1, random_state = random_seed)

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
