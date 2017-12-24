import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x = pd.read_csv('x.csv')
y = pd.read_csv('y.csv')
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=20)
test = pd.read_csv('testing.csv')
testid = pd.read_csv('test.csv')
testid = testid['User_ID']
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def keras_classifier(algo='adam',init_mode='uniform',activfun='relu',neurons=128):
    classifier = Sequential()

    classifier.add(Dense(units = neurons, kernel_initializer = init_mode, activation = activfun, input_dim = 508))
    classifier.add(Dropout(rate = 0.4))

    classifier.add(Dense(units = neurons, kernel_initializer = init_mode, activation = activfun))
    classifier.add(Dropout( rate = 0.4))
    
    classifier.add(Dense(units = neurons, kernel_initializer = init_mode, activation = activfun))
    classifier.add(Dropout(rate= 0.4))

    classifier.add(Dense(units = 1, kernel_initializer = init_mode, activation = 'sigmoid'))
    classifier.add(Dropout(rate= 0.4))
    
    classifier.compile(optimizer = algo, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
batch_size = [500]
epochs = [20,30]
neurons = [254,512,1024]
dropout_rate = [0.4, 0.6, 0.7, 0.9]
optimizer = ['SGD', 'Adam', 'Adamax', 'Nadam']
init_mode = ['uniform',  'glorot_uniform', 'he_normal']
activation = ['softmax', 'relu', 'tanh', 'sigmoid']
model = KerasClassifier(build_fn=keras_classifier, verbose=0)    
param_grid = dict(batch_size=batch_size, epochs=epochs,neurons=neurons, algo=optimizer,init_mode=init_mode,activfun=activation)       
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,verbose=1,cv=8)
grid_result = grid.fit(np.array(x_train), np.array(y_train))
print(grid_result.best_score_,grid_result.best_params_)
preds = grid_result.predict(np.array(x_test))
y_pred = np.where(preds >= 0.01,1,0)
score = accuracy_score(y_test,y_pred)
print('ONTEST',score)
