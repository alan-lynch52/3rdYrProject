import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
#test = pd.read_csv("test.csv")

y = train["label"]
x = train.drop(labels = ["label"], axis = 1)

del train

print(y)

print(x)
random_seed = 2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = random_seed)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

print(rf.predict(x_test))


