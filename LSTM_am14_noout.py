
"""


Author: Adi Joshi, Dylan Hematillake
Date : 3/26/2018/

Purpose: To test the NBM model on turbine 
        with a gearbox repalcement 
Input: Preprocessed SCADA data for 2 turbines,2 date vectors,2 reseason vectors,
        NBM weights
Output: Interactive HTML file of the NBM for gearbox temperature on the training and 
        testing turbine using Bokeh
Imports,Libraries Used: Pandas,Numpy,Keras,Matplotlib,bokeh 
Dependencies: 
    fetch_data:To split data into the training and shape it into 3D shape for 
    LSTM prediction
    reshape: Reshape data into 2D array after data has been modeled. 
    rescale_data: reseason and rescale data for visualization and make outliers
    values NA
    

"""


import numpy as np

import os
import warnings




from sklearn.metrics import mean_squared_error
from math import sqrt

from pandas import read_csv
from matplotlib import pyplot

import pandas as pd

from bokeh.io import output_file,show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from keras.models import model_from_json
from bokeh.models import Range1d
from bokeh.models import HoverTool

from datetime import date
from random import randint

from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn

#Purpose to shape the data in training and testing datatset
def fetch_data (dataset, n_train_hours):
     
     values = dataset.values
     train = values[:n_train_hours, :]
     test = values[n_train_hours: , :]
     train_X, train_y = train[:, :-1], train[:, -1]
     test_X, test_y = test[:, :-1], test[:, -1]
     train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
     test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
     return (train_X,train_y,test_X,test_y);
 
#Change array shape after modeling
def reshape (train_X,train_y):

    train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
    train_y = train_y.reshape((len(train_y), 1))
    concate_train_y = np.concatenate((train_y, train_X[:, 1:]), axis=1)
    y_train = concate_train_y[:,0] 
    
    return(y_train);   
    
#Rescale data inorder for visualization,
#scale and season variables imported from R  
def rescale_data (pred,scale,center,season,thres):
    pred[pred < thres] = np.nan
    pred = (pred*scale)+center
    pred = np.add(pred,season)
    return (pred);

#AM14 and AM69 files have been preprocessed in R
dataset_am14 = read_csv('C:\\Users\\Admin\\Documents\\SCADA\\am14_seasonal.txt',
                        header=0, index_col=None, sep="\t")
dataset_am69 = read_csv('C:\\Users\\Admin\\Documents\\SCADA\\am69_seasonal.txt', 
                        header=0, index_col=None, sep="\t")

#reason vales taken from preprocessed values in R
reseason_am14 = read_csv('C:\\Users\\Admin\\Documents\\SCADA\\reseason_am14_gbox.txt', 
                         header=0, index_col=None, sep="\t")
reseason_am69 = read_csv('C:\\Users\\Admin\\Documents\\SCADA\\reseason_am69_gbox.txt',
                         header=0, index_col=None, sep="\t")
#date index taken from R.Index required for visuals in bokeh and matplotlib 
dat14 = read_csv('C:\\Users\\Admin\\Documents\\SCADA\\date14.txt',
                 header=0, index_col=None, sep="\t")

dat69 = read_csv('C:\\Users\\Admin\\Documents\\SCADA\\date69.txt', 
                 header=0, index_col=None, sep="\t")

#Changing shape of datetime
dat14 = pd.to_datetime(dat14.stack()).unstack()
dat69 = pd.to_datetime(dat69.stack()).unstack()

dat14 = dat14.values.ravel()
dat69 = dat69.values.ravel()

val_am14 = reseason_am14.values
val_am14 = val_am14.ravel()

val_am69 = reseason_am69.values
val_am69 = val_am69.ravel()




am14_xTrain,am14_yTrain,am14_xTest,am14_yTest = fetch_data(dataset_am14,159262)
am69_xTrain,am69_yTrain,am69_xTest,am69_yTest = fetch_data(dataset_am69,156079)

#reading model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model_noout_v2.h5")

am14Pred = model.predict(am14_xTrain)
am69Pred = model.predict(am69_xTrain)


am14_yTrain = reshape(am14_xTrain,am14_yTrain)
am69_yTrain = reshape(am69_xTrain,am69_yTrain)
am69Pred = am69Pred.ravel()
am14Pred = am14Pred.ravel()

rmse_am14 = sqrt(mean_squared_error(am14_yTrain,am14Pred ))
print('AM14 RMSE: %.3f' % rmse_am14)

rmse_am69 = sqrt(mean_squared_error(am69_yTrain,am69Pred ))
print('AM69 RMSE: %.3f' % rmse_am69)


   
#rescale and seasonalize data for visualization    
am69_yTrain = rescale_data (am69_yTrain,11.08612897,60.0120437,val_am69,-3)
am69Pred = rescale_data (am69Pred,11.08612897,60.0120437,val_am69,-3)

am14_yTrain = rescale_data(am14_yTrain,14.74854,57.60717,val_am14,-3)
am14Pred = rescale_data(am14Pred,14.74854,57.60717,val_am14,-3)

dif69 =  np.subtract (am69_yTrain,am69Pred) #creating a residual vector
dif14 =  np.subtract (am14_yTrain,am14Pred)

#matplot
pyplot.plot(dat14,dif14,color="green",label ="Residual",alpha =0.5)
pyplot.plot(dat14,am14Pred,color="#0000FF",label ="Predicted")
pyplot.plot(dat14,am14_yTrain,color="red",label ="Predicted",alpha =0.5)          
pyplot.legend()
pyplot.show()

#Bokeh figures          
fig1 = figure(title="AM14",
             tools="pan,box_zoom,wheel_zoom,reset,box_select,lasso_select,save",
             x_axis_type="datetime")
fig1.border_fill_color = "whitesmoke"
fig1.min_border_right = 30

df14= pd.DataFrame({'train': am14_yTrain,'pred':am14Pred,'dif':dif14}, 
                   index = dat14) 

fig1.line(df14.index,df14['train'],
          legend="SCADA Data",
          color="blue",muted_color="blue", muted_alpha=0.2)
fig1.line(df14.index,df14['pred'],
          legend='Modeled Data',
          color="red",muted_color="red", muted_alpha=0.2)
fig1.line(df14.index,df14['dif'],
          legend='Difference',
          color="green",muted_color="green", muted_alpha=0.0)

fig1.legend.location="bottom_right"
fig1.legend.click_policy="mute"
fig1.xaxis.axis_label = "Time (10 Mins)"
fig1.yaxis.axis_label = "Gearbox Temp (Deg C)"
fig1.yaxis.major_label_orientation = "vertical"
bottom, top =  -20, 90
fig1.y_range = Range1d(bottom, top)


df69= pd.DataFrame({'train': am69_yTrain,'pred':am69Pred,'dif':dif69}, 
                   index = dat69)



fig2 = figure(title='AM69',
             tools="pan,box_zoom,wheel_zoom,reset,box_select,save",
             x_axis_type="datetime")
fig2.border_fill_color = "whitesmoke"
fig2.min_border_right = 30
fig2.line(df69.index,df69['train'],
          legend="SCADA Data",
          color="blue",muted_color="blue", muted_alpha=0.2)
fig2.line(df69.index,df69['pred'],
          legend='Modeled Data',
          color="red",muted_color="red", muted_alpha=0.2)
fig2.line(df69.index,df69['dif'],
          legend='Difference',
          color="green",muted_color="green", muted_alpha=0.0)

fig2.legend.location ="bottom_right"
fig2.legend.click_policy="mute"
fig2.xaxis.axis_label = "Time (10 Mins)"
fig2.yaxis.axis_label = "Gearbox Temp (Deg C)"
fig2.yaxis.major_label_orientation = "vertical"
bottom, top =  -20, 90
fig2.y_range = Range1d(bottom, top)

p=gridplot([[fig2,fig1]])
output_file('NBM_Plots_noout_v2.html')
show (p)







