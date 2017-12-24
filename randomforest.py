#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:04:08 2017

@author: anirudhmuhnot
"""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('train.csv')
param_grid = {"max_depth": [3,4,6, None],
                            "n_estimators": [300,400,500],
                          "max_features": [1,2,4,'auto', 'sqrt', 'log2'],
                           "min_samples_split": [4,5,6,8,10],
                            "min_samples_leaf": [1,2,4,5,7],
                           "criterion": ["gini", "entropy"]}

clf = RandomForestClassifier(oob_score = True)
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10, verbose=1, n_jobs=-1)
clf.fit(train,y)
grid_result = grid_search.fit(train, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
