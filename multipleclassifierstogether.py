import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#loading datatset
x = pd.read_csv('x.csv')
y = pd.read_csv('y.csv')
#test = pd.read_csv('testing.csv')
class find_params:
	def __init__ (self,models):
		self.models = models
		self.keys = models.keys()
	def find_all(self):
		for key in self.keys:
			clf = self.models[key]
			print("For model",key)
			print(clf.get_params().keys())

class model_selection:
	def __init__(self, models,parameters,test_size=0.2):
		if not set(models.keys()).issubset(set(parameters.keys())):
			print("Models are missing paramaters",(model.keys()-parameters.keys()))
		self.models = models
		self.parameters = parameters 
		self.keys = models.keys()

	def fit_test(self,x_train,x_test,y_train,y_test,cv=3,verbose=0, scoring=None, n_jobs=-1):
		for key in self.keys:
			model = self.models[key]
			parameters = self.parameters[key]
			grid_search = GridSearchCV(model,param_grid=parameters,n_jobs=1, cv=cv,verbose=verbose,scoring=scoring)
			print("Fitting model",key)
			grid_search.fit(x_train,y_train.reshape(31145))
			print('For model',key, 'best score is',grid_search.best_score_,'using the parameters',grid_search.best_params_)
			print("Predicting for testing dataset ..")
			preds = grid_search.predict(x_test)
			score = accuracy_score(y_test.reshape(7787),preds)
			print('ON TESTING DATASET, For model',key, 'best score is',score,"after gird search.")
			print("NEXT MODEL TUNING..")


models1 = { 
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
}

params1 = { 
    'ExtraTreesClassifier': {"n_estimators": [20,30,50],
              "max_depth": [3,8,15],
              "min_samples_split": [20,50,100],
              "min_samples_leaf": [10,20,50],
              "max_leaf_nodes": [20,40,10],
              "min_weight_fraction_leaf": [0.1,0.2]},
    'RandomForestClassifier': {"n_estimators": [20,30,50],
              "max_depth": [3,8,15],
              "min_samples_split": [20,50,100],
              "min_samples_leaf": [10,20,50],
              "max_leaf_nodes": [20,40,10],
              "min_weight_fraction_leaf": [0.1,0.2]},
    'GradientBoostingClassifier': {"n_estimators": [20,30,50],
              "max_depth": [3,8,15],
              "min_samples_split": [20,50,100],
              "min_samples_leaf": [10,20,50],
              "max_leaf_nodes": [20,40,10],
              "min_weight_fraction_leaf": [0.1,0.2],
              "learning_rate": [0.2,0.5,0.8]
              }
    # 'SVC': [
    #     {'kernel': ['linear'], 'C': [1, 10]},
    #     {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    # ]
}
#parameter_list = find_params(models1)
#parameter_list.find_all()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
classifier = model_selection(models1,params1)
classifier.fit_test(np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test))