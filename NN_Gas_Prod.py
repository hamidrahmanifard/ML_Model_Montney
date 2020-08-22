# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:51:57 2020

@author: hamid.rahmanifard
"""

# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import os
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn import preprocessing
from keras.layers import Dropout
from keras.constraints import maxnorm
from sklearn.model_selection import GridSearchCV


os.chdir('C:/Users/hamid.rahmanifard/OneDrive - University of Calgary/UofC/PhD/Publication/Journal/Montney gas prod/BC/Python/Montney')

# coefficient of determination (R^2) for regression  (only for Keras tensors)
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Function to create model, required for KerasClassifier
def create_model(optimizer= 'rmsprop' , init= 'glorot_uniform'):#, dropout_rate=0.0, weight_constraint=0):
    # create model
    model = Sequential()
    #model.add(Dropout(dropout_rate))
    model.add(Dense(40, input_dim=11, activation='relu', kernel_initializer=init))#, kernel_constraint=maxnorm(weight_constraint)))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(40, kernel_initializer= init , activation= 'relu'))#, kernel_constraint=maxnorm(weight_constraint)))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(40, kernel_initializer= init , activation= 'relu'))#, kernel_constraint=maxnorm(weight_constraint)))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    # Compile model
    model.compile(loss= 'mean_squared_error' , optimizer=optimizer, metrics=[r_square])
    return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
dataframe = pd.read_csv("Final_db1.csv", header=0)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:11]
Y = dataset[:,11]

# split into two groups
X1,X2,Y1,Y2=train_test_split(X,Y,test_size=0.05)
# create model
model = KerasRegressor(build_fn=create_model, verbose=0)

# grid search epochs, batch size and optimizer
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#init = ['uniform', 'lecun_uniform', 'normal', 'glorot_normal', 'glorot_uniform']
epochs = [150, 300, 600, 1200]
batches = [5, 10, 20, 40]
#weight_constraint = [1, 2, 3, 4, 5]
#dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


param_grid = dict(optimizer=optimizer, nb_epoch=epochs, batch_size=batches)#, init=init,
                 # weight_constraint=weight_constraint,dropout_rate=dropout_rate )

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=10)
grid_result = grid.fit(X, Y)
# Each combination is then evaluated using the default of 3-fold stratified cross validation.

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))



means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))




'''
print(grid.best_params_)
grid.best_estimator_.model.save('filename.h5')

from sklearn.externals import joblib
joblib.dump(grid.best_estimator_, '123.pkl')
'''