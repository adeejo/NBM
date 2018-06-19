"""
Author: Adi Joshi, Dylan Hematillake
Date : 3/26/2018/

Purpose: To test the NBM model on turbine 
        with a gearbox replacement 
Input: Preprocessed SCADA data for 2 turbines,2 date vectors,2 reseason vectors,
        NBM weights
Output: Interactive HTML file of the NBM for gearbox temperature on the training 
and testing turbine using Bokeh library
Imports,Libraries Used: Pandas,Numpy,Keras,Matplotlib,bokeh 
Dependencies: 
    fetch_data:To split data into the training and shape it into 3D shape for 
    LSTM prediction
    reshape: Reshape data into 2D array after data has been modeled. 
    rescale_data: reseason and rescale data for visualization and make outliers
    values NA
   
"""


from NBM_Train import shape_before_model,shape_after_model

from keras.models import load_model
from sklearn.metrics import mean_squared_error

from math import sqrt
import numpy as np
import pandas as pd

from bokeh.io import output_file,show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import Range1d

#Rescale data inorder for visualization,
#scale and season variables imported from R  
def rescale_data (pred,scale,center,season,thres):
    pred[pred < thres] = np.nan
    pred = (pred*scale)+center
    pred = np.add(pred,season)
    return (pred);

#Bokeh figures
def plot(yTrain,Pred,diff,data,title):        
    fig = figure(title=title,
                 tools="pan,box_zoom,wheel_zoom,reset,box_select,lasso_select,save",
                 x_axis_type="datetime")
    fig.border_fill_color = "whitesmoke"
    fig.min_border_right = 30
    
    df14= pd.DataFrame({'train': yTrain,'pred':Pred,'dif':diff}, 
                       index = data) 
    
    fig.line(df14.index,df14['train'],
              legend="SCADA Data",
              color="blue",muted_color="blue", muted_alpha=0.2)
    fig.line(df14.index,df14['pred'],
              legend='Modeled Data',
              color="red",muted_color="red", muted_alpha=0.2)
    fig.line(df14.index,df14['dif'],
              legend='Difference',
              color="green",muted_color="green", muted_alpha=0.0)
    
    fig.legend.location="bottom_right"
    fig.legend.click_policy="mute"
    fig.xaxis.axis_label = "Time (10 Mins)"
    fig.yaxis.axis_label = "Gearbox Temp (Deg C)"
    fig.yaxis.major_label_orientation = "vertical"
    bottom, top =  -20, 90
    fig.y_range = Range1d(bottom, top)
    return fig

def main():

    #AM14 and AM69 files have been preprocessed in R
    dataset_am14 = pd.read_csv('am14_seasonal.txt',
                            header=0, index_col=None, sep="\t")
    dataset_am69 = pd.read_csv('am69_seasonal.txt', 
                            header=0, index_col=None, sep="\t")
    
    #reseason values taken from preprocessed values in R
    reseason_am14 = pd.read_csv('reseason_am14_gbox.txt', 
                             header=0, index_col=None, sep="\t")
    reseason_am69 = pd.read_csv('reseason_am69_gbox.txt',
                             header=0, index_col=None, sep="\t")
    #date index taken from R.Index required for visuals in bokeh and matplotlib 
    dat14 = pd.read_csv('date14.txt',
                     header=0, index_col=None, sep="\t")
    dat69 = pd.read_csv('date69.txt', 
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
    
    #Reshape data before modeling
    am14_xTrain,am14_yTrain,am14_xTest,am14_yTest = shape_before_model(dataset_am14,
                                                               159262)
    am69_xTrain,am69_yTrain,am69_xTest,am69_yTest = shape_before_model(dataset_am69,
                                                               156079)
    #Running the model
    model = load_model("model_noout_v2.h5")
    
    am14Pred = model.predict(am14_xTrain)
    am69Pred = model.predict(am69_xTrain)
    
    #Reshape data after modeling
    am14_yTrain = shape_after_model(am14_xTrain,am14_yTrain)
    am69_yTrain = shape_after_model(am69_xTrain,am69_yTrain)
    am69Pred = am69Pred.ravel()
    am14Pred = am14Pred.ravel()
    
    #Calculating root mean squared error btw actual and predicted
    rmse_am14 = sqrt(mean_squared_error(am14_yTrain,am14Pred ))
    print('AM14 RMSE: %.3f' % rmse_am14)
    
    rmse_am69 = sqrt(mean_squared_error(am69_yTrain,am69Pred ))
    print('AM69 RMSE: %.3f' % rmse_am69)
       
    #rescale,reseasonalize,make outliers nan for data for visualization    
    am69_yTrain = rescale_data(am69_yTrain,11.08612897,60.0120437,val_am69,-3)
    am69Pred = rescale_data(am69Pred,11.08612897,60.0120437,val_am69,-3)
    
    am14_yTrain = rescale_data(am14_yTrain,14.74854,57.60717,val_am14,-3)
    am14Pred = rescale_data(am14Pred,14.74854,57.60717,val_am14,-3)
    
    dif69 =  np.subtract(am69_yTrain,am69Pred) #creating a residual vector
    dif14 =  np.subtract(am14_yTrain,am14Pred)
    
    fig1 = plot(am14_yTrain,am14Pred,dif14,dat14,'AM14')   
    fig2 = plot(am69_yTrain,am69Pred,dif69,dat69,'AM69')   
    
    p = gridplot([[fig2,fig1]])
    output_file('NBM_Plots_noout_v2.html')
    show(p)

if __name__ == '__main__':
    main()
