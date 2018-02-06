# -*- coding: utf-8 -*-
"""
Program Intended to perform modeling after data-preprocessing
"""

# theano
import theano
print('theano: %s' % theano.__version__)
# tensorflow
import tensorflow
print('tensorflow: %s' % tensorflow.__version__)
# keras
import keras
print('keras: %s' % keras.__version__)




from keras.layers.core import Dense, Activation, Dropout

import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.recurrent import LSTM
from keras.models import Sequential
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

from sklearn.metrics import mean_squared_error
from math import sqrt

from pandas import read_csv
from matplotlib import pyplot
from keras.layers import Masking

dataset = read_csv('C:\\Users\\Admin\\Documents\\SCADA\\all_var_1000_noOut.txt', header=0, index_col=0, sep="\t")
n_train_hours = 700
values = dataset.values
#values[values < 0] = -1 # make all negative values -1

train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
train_X, train_y = train[:, 2:], train[:, :1]#output variable is the first column
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1])) #reshape the data to have input,timestamp, outputshape
   


test_X, test_y = test[:, 2:], test[:, :1]
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

model = Sequential()
model.add(Masking(mask_value=-4, input_shape=(train_X.shape[1], 10) ))

model.add(LSTM(50, input_shape=(train_X.shape[1], 10),return_sequences = True)) # Number sample/alpha(Number of )
model.add(LSTM(25,return_sequences = True))
model.add(LSTM(10))

model.add(Dense(1)) # output shape has 1 dense output layer. A dense layer thus is used to change the dimensions of your vector. 
# epoch is one pass over the entire dataset
model.compile(loss='mae', optimizer='adam')
history = model.fit(train_X, train_y, epochs=50, batch_size=100, validation_data=(test_X, test_y), verbose=2, shuffle=False)

xy = model.predict(train_X)
yhat = model.predict(test_X,batch_size= 20)

#yhat
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))

inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)

inv_yhat = inv_yhat[:,0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)

train_y = train_y.reshape((len(train_y), 1))
invt_y = np.concatenate((train_y, train_X[:, 1:]), axis=1)


inv_y = inv_y[:,0]
invt_y = invt_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, yhat))
print('Test RMSE: %.3f' % rmse)

rmses = sqrt(mean_squared_error(invt_y, xy))
print('Train RMSE: %.3f' % rmses)
plt.ylim( 50, 75 ) 
pyplot.plot(yhat*4.976601+62.6154191,color="blue",label ="Predicted")
pyplot.plot(inv_y*4.976601+62.6154191,color="red",label ="Actual")
pyplot.legend()
pyplot.show()
plt.ylim( 50, 75 ) 
pyplot.plot(xy*4.976601+62.6154191,color ="blue",label="Predicted")
pyplot.plot(train_y*4.976601+62.6154191,color="red",label ="Actual")
pyplot.legend()
pyplot.show()

