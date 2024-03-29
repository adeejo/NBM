

"""
Author: Adi Joshi, Dylan Hematillake 
Date : 3/26/2018/
    
    
Program Intended to perform LSTM time series modeling after data has been 
preprocessed for a SCADA data for a wind turbine

Input: Preprocessed SCADA data 
Output: RNN Weights and training testing plots

Imports,Libraries Used: Pandas,Numpy,Keras,Matplotlib

Dependencies
shape_before_model: To split data into the training and testing data and shape it into 
            3D shape for LSTM modeling.
LSTM_model: To create a LSTM model with 3 layers using keras library.
reshape: Reshape data into 2D array after data has been modeled. 
"""

from keras.layers.core import Dense 
from keras.layers.recurrent import LSTM 
from keras.models import Sequential

from sklearn.metrics import mean_squared_error

from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

#To get the training and testing data in 3D shape for LSTM
def shape_before_model(dataset, N_TRAIN_HOURS):

     values = dataset.values
     train = values[:N_TRAIN_HOURS, :]
     test = values[N_TRAIN_HOURS: , :]
     train_X, train_y = train[:, :-1], train[:, -1]
     test_X, test_y = test[:, :-1], test[:, -1]
     train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
     test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
     return (train_X,train_y,test_X,test_y); 

 
#To create a LSTM model with 3 layers using keras library. 
def LSTM_model(shape1,shape2,w1,w2,w3):
    
    model = Sequential()

    model.add(LSTM(w1,input_shape=(shape1,shape2),return_sequences = True)) 
    model.add(LSTM(w2,return_sequences = True))
    model.add(LSTM(w3))
    model.add(Dense(1)) 
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return (model); 

#To Reshape data into 2D after completing modeling 
def shape_after_model(train_X,train_y):

    train_X = train_X.reshape((train_X.shape[0],train_X.shape[2]))
    train_y = train_y.reshape((len(train_y), 1))
    concate_train_y = np.concatenate((train_y, train_X[:, 1:]),axis=1)

    y_train = concate_train_y[:,0] 

    return(y_train);

#fitting the model
def fit_model(model,train_X,train_y,test_X,test_y):
    model.fit(train_X, train_y, epochs=20, batch_size=100, 
                        validation_data=(test_X, test_y), 
                        verbose=2, shuffle=False)
    yhat_train = model.predict(train_X)
    yhat_test = model.predict(test_X,batch_size= 20)
    
    y_train = shape_after_model(train_X,train_y)
    y_test = shape_after_model(test_X,test_y)
    return yhat_train,yhat_test,y_train,y_test
    

#ploting training and testing using matplotlib
def plot(y_train,yhat_train,y_test,yhat_test):
    plt.ylim( -4, 4 ) 
    plt.plot((y_train),color="yellow",label ="Actual",alpha = 0.5) #training set
    plt.plot((yhat_train),color ="blue",label="Modeled")
    plt.legend()
    plt.show()
    
    plt.ylim( -4, 4 ) 
    plt.plot((y_test),color="yellow",label ="Actual",alpha = 0.5) #testing set
    plt.plot((yhat_test),color="blue",label ="Predicted")
    plt.legend()
    plt.show()
    return;
    
def main():
    dataset = read_csv('am69_seasonal.txt', 
                       header=0, index_col=None, sep="\t")
                      

    N_TRAIN_HOURS = 109256
    train_X,train_y,test_X,test_y = shape_before_model(dataset,N_TRAIN_HOURS)
    #Creating model
    model = LSTM_model(train_X.shape[1],train_X.shape[2], 50,25,10) 
    
    yhat_train,yhat_test,y_train,y_test = fit_model(model,
                                                  train_X,train_y,
                                                  test_X,test_y)
    
    #Calculating root mean squared error btw actual and predicted
    rmses = sqrt(mean_squared_error(y_train, yhat_train))
    print('Train RMSE: %.3f' % rmses)
    
    rmse = sqrt(mean_squared_error(y_test, yhat_test))
    print('Test RMSE: %.3f' % rmse)

    #Saving model 
    model.save("model_noout_v2.h5")
    print("Saved model to disk")
    
    plot(y_train,yhat_train,y_test,yhat_test)

if __name__ == "__main__":
    main()
