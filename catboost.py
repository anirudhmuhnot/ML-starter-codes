import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

df = pd.read_csv('train.csv')

tuned_parameters = {'iterations': [10, 20, 30, 50], 'learning_rate': [0.01, 0.03, 0.1], 'depth': [4, 6, 8]}
score = 'roc_auc'
print()
from catboost import CatBosstClassifier
clf = GridSearchCV(CatBosstClassifier(), tuned_parameters, cv=3, scoring=score)
clf.fit(df, target)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
clf.save_model('saved_model')