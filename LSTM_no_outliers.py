
# -*- coding: utf-8 -*-
"""
Author: Adi Joshi 
Date : 3/26/2018/
    
    
Program Intended to perform LSTM time series modeling after data has been 
preprocessed for a SCADA data for a wind turbine

Input: Preprocessed SCADA data 
Output: RNN Weights and training testing plots

Imports,Libraries Used: Pandas,Numpy,Keras,Matplotlib

Dependencies
fetch_data: To split data into the training and testing data and shape it into 
            3D shape for LSTM modeling.
LSTM_model: To create a LSTM model with 3 layers using keras library.
reshape: Reshape data into 2D array after data has been modeled. 
"""

import theano

import tensorflow

import keras

from keras.layers.core import Dense, Activation, Dropout

import os

import warnings
import numpy as np
from keras.layers.recurrent import LSTM
from keras.models import Sequential
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

from sklearn.metrics import mean_squared_error
from math import sqrt

from pandas import read_csv
from matplotlib import pyplot
from keras.layers import Masking


#To get the training and testing data in 3D shape for LSTM
def fetch_data (dataset, n_train_hours):
     
     values = dataset.values
     train = values[:n_train_hours, :]
     test = values[n_train_hours: , :]
     train_X, train_y = train[:, :-1], train[:, -1]
     test_X, test_y = test[:, :-1], test[:, -1]
     train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
     test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
     return (train_X,train_y,test_X,test_y); 
 
#To create a LSTM model with 3 layers using keras library. 
def LSTM_model (shape1,shape2,w1,w2,w3):
    model = Sequential()

    model.add(LSTM(w1, input_shape=(shape1,shape2 ),
                   return_sequences = True)) 
    model.add(LSTM(w2,return_sequences = True))
    model.add(LSTM(w3))

    model.add(Dense(1)) 
    model.compile(loss='mean_squared_error', optimizer='adam')
    return (model); 


#To Reshape data into 2D after completing modeling 
def reshape (train_X,train_y):

    train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
    train_y = train_y.reshape((len(train_y), 1))
    concate_train_y = np.concatenate((train_y, train_X[:, 1:]), axis=1)
    y_train = concate_train_y[:,0] 
    
    return(y_train);
    


dataset = read_csv('C:\\Users\\Admin\\Documents\\SCADA\\am69_seasonal.txt', 
                   header=0, index_col=None, sep="\t")

n_train_hours = 109256
train_X,train_y,test_X,test_y=fetch_data(dataset,n_train_hours)

model = LSTM_model(train_X.shape[1],train_X.shape[2], 50,25,10)

#fitting the model
history = model.fit(train_X, train_y, epochs=20, batch_size=100, 
                    validation_data=(test_X, test_y), verbose=2, shuffle=False)

yhat_train = model.predict(train_X)
yhat_test = model.predict(test_X,batch_size= 20)

    
y_train = reshape(train_X,train_y)
y_test = reshape(test_X,test_y)


#calculating root measn squred error btw actual and predicted
rmses = sqrt(mean_squared_error(y_train, yhat_train))
print('Train RMSE: %.3f' % rmses)

rmse = sqrt(mean_squared_error(y_test, yhat_test))
print('Test RMSE: %.3f' % rmse)



#saving model weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_noout_v2.h5")
print("Saved model to disk")

#ploting training and testing using matplot.lib
pyplot.ylim( -4, 4 ) 
pyplot.plot((y_train),color="yellow",label ="Actual",alpha = 0.5) #training set
pyplot.plot((yhat_train),color ="blue",label="Modeled")
pyplot.legend()
pyplot.show()

pyplot.ylim( -4, 4 ) 
pyplot.plot((y_test),color="yellow",label ="Actual",alpha = 0.5) #testing set
pyplot.plot((yhat_test),color="blue",label ="Predicted")
pyplot.legend()
pyplot.show()



















