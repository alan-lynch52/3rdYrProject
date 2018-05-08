import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
y = train[LABELS]
y['clean'] = (train[LABELS].sum(axis=1)==0)
c_dist = y.sum()
print(c_dist)

x = np.arange(7)
plt.bar(x, c_dist)
plt.xticks(x,['toxic','sev toxic','obscene','threat','insult','id hate','clean'], rotation=30)
plt.ylabel("Frequency")
plt.xlabel("Labels")
plt.show()
