from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
estimator = AdaBoostClassifier(n_estimators=200)
params = {
'n_estimators':np.linspace(50,500,num=10),
'learning_rate': np.linspace(0.1,1,num=10)	
}
grid_search = GridSearchCV(estimator,param_grid=params,n_jobs=-1, cv=5,verbose=1,scoring='accuracy')
grid_search = grid_search.fit(np.array(x), np.array(y))
print('For model adaboost', 'best score is',grid_search.best_score_,'using the parameters',grid_search.best_params_)