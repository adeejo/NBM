# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:41:25 2018

@author: Admin
"""

from keras.models import load_model
import numpy as np
from numpy import newaxis
import os
import warnings
from keras.models import Sequential
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

from sklearn.metrics import mean_squared_error
from math import sqrt

from pandas import read_csv
from matplotlib import pyplot


from bokeh.io import output_file,show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from keras.models import model_from_json
from bokeh.models import Range1d

def fetch_data (dataset, n_train_hours):
     
     values = dataset.values
     train = values[:n_train_hours, :]
     test = values[n_train_hours: , :]
     train_X, train_y = train[:, :-1], train[:, -1]
     test_X, test_y = test[:, :-1], test[:, -1]
     train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
     test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
     return (train_X,train_y,test_X,test_y);
 

def reshape (train_X,train_y):

    train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
    train_y = train_y.reshape((len(train_y), 1))
    concate_train_y = np.concatenate((train_y, train_X[:, 1:]), axis=1)
    y_train = concate_train_y[:,0] #variable for calculating RMSE
    
    return(y_train);    

dataset_am14 = read_csv('C:\\Users\\Admin\\Documents\\SCADA\\am14_outliers_seasonal.txt', header=0, index_col=None, sep="\t")
dataset_am69 = read_csv('C:\\Users\\Admin\\Documents\\SCADA\\am69_outliers_seasonal.txt', header=0, index_col=None, sep="\t")


reseason_am14 = read_csv('C:\\Users\\Admin\\Documents\\LSTM_GIT\\reseason_am14.txt', header=0, index_col=None, sep="\t")
reseason_am69 = read_csv('C:\\Users\\Admin\\Documents\\LSTM_GIT\\reseason_am69.txt', header=0, index_col=None, sep="\t")

val_am14 = reseason_am14.values
val_am14 = val_am14.ravel()

val_am69 = reseason_am69.values
val_am69 = val_am69.ravel()




am14_xTrain,am14_yTrain,am14_xTest,am14_yTest = fetch_data(dataset_am14,159262)
am69_xTrain,am69_yTrain,am69_xTest,am69_yTest = fetch_data(dataset_am69,156081)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")


am14Pred = model.predict(am14_xTrain)
am69Pred = model.predict(am69_xTrain)


am14_yTrain = reshape(am14_xTrain,am14_yTrain)
am69_yTrain = reshape(am69_xTrain,am69_yTrain)

rmse_am14 = sqrt(mean_squared_error(am14_yTrain,am14Pred ))
print('AM14 RMSE: %.3f' % rmse_am14)

rmse_am69 = sqrt(mean_squared_error(am69_yTrain,am69Pred ))
print('AM69 RMSE: %.3f' % rmse_am69)

am69_yTrain = (am69_yTrain*15.74945492)+58.3749263
am69_yTrain = np.add(am69_yTrain,val_am69)

am69Pred = (am69Pred*15.74945492)+58.3749263
am69Pred = am69Pred.ravel()
am69Pred = np.add(am69Pred,val_am69) #reseason

am14_yTrain = (am14_yTrain*17.22179)+56.19851
am14_yTrain = np.add(am14_yTrain,val_am14)

am14Pred = (am14Pred*17.22179)+56.19851
am14Pred = am14Pred.ravel()
am14Pred = np.add(am14Pred,val_am14) #reseason



fig1 = figure(title="AM14",
             tools="pan,box_zoom,wheel_zoom,reset")
fig1.border_fill_color = "whitesmoke"
fig1.min_border_right = 30

#xTrain=yhat_train;

xTrain_am14 = range(0,159262,1);
#y_train =(y_train*1)+0;
fig1.line(xTrain_am14,am14_yTrain,legend="SCADA Data",color="blue")
fig1.line(xTrain_am14,am14Pred,color="red",legend='Predicted')
fig1.legend.location="bottom_right"
fig1.xaxis.axis_label = "Time (10 Mins)"
fig1.yaxis.axis_label = "Temperature (Deg C)"
fig1.yaxis.major_label_orientation = "vertical"
bottom, top =  10, 90
fig1.y_range = Range1d(bottom, top)

xTrain_am69 = range(0,156081,1);
fig2 = figure(title='AM69',
             tools="pan,box_zoom,wheel_zoom,reset")
fig2.border_fill_color = "whitesmoke"

fig2.line(xTrain_am69,am69_yTrain,legend="SCADA Data",color="blue")
fig2.line(xTrain_am69,am69Pred,color="red",legend='Predicted')
fig2.legend.location="bottom_right"
fig2.xaxis.axis_label = "Time (10 Mins)"
fig2.yaxis.axis_label = "Temperature (Deg C)"
fig2.yaxis.major_label_orientation = "vertical"
bottom, top =  10, 90
fig2.y_range = Range1d(bottom, top)


p=gridplot([[fig2,fig1]])
output_file('NBM_Plots.html')
show(p)








