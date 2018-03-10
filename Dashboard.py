from NBM_DH import fetch_data,estimate
from bokeh.io import output_file,show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
import pandas as pd
from keras.models import load_model

dataset = pd.read_csv('D:/all_var_1000_v6.txt', header=0, index_col=0, sep="\t")
train_X,train_y,test_X,test_y=fetch_data(dataset,700)

model = load_model('NBM.h5')

xy = model.predict(train_X)
yhat = model.predict(test_X,batch_size= 20)

yhat,inv_y,xy,invt_y,xTrain,xTest=estimate(test_X,train_X,yhat,xy,test_y,train_y)
     
fig1 = figure(title="Testing")
fig1.border_fill_color = "whitesmoke"
fig1.min_border_right = 30

fig1.line(xTrain,inv_y,legend="SCADA Data")
fig1.line(xTrain,yhat,color="red",legend='Predicted')
fig1.legend.location="bottom_right"
fig1.xaxis.axis_label = "Time (10 Mins)"
fig1.yaxis.axis_label = "Temperature (Deg C)"
fig1.yaxis.major_label_orientation = "vertical"



fig2 = figure(title='Training')
fig2.border_fill_color = "whitesmoke"
fig2.line(xTest,xy,color="red",legend='Predicted') 
fig2.line(xTest,invt_y,legend="SCADA Data")
fig2.yaxis.axis_label = "Temperature (Deg C)"
fig2.yaxis.major_label_orientation = "vertical"
fig2.xaxis.axis_label = "Time (10 Mins)"

p=gridplot([[fig2,fig1]])
output_file('NBM_Plots.html')
show(p)
