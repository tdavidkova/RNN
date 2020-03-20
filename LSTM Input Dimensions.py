#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:35:20 2020

@author: tdavidkova
"""
#### Build NN LSTM model for prediction of X outcomes ahead


import numpy as np
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib 
import matplotlib.pyplot as plt
import numpy
from numpy import concatenate

# fix random seed for reproducibility
numpy.random.seed(7)

#### Model without exogeneous variables, see below general case - exogeneous variables

x = np.array([i for i in range(1,37)])
y = np.array([i for i in range(37,40)])


def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	return df

timesteps = 3

x1 = timeseries_to_supervised(x,timesteps)
x1
x1 = x1[timesteps:]
x1
x2 = np.array(x1)
x2
x3 = x2[:,:timesteps]
x3


x4 = x3.reshape(x3.shape[0], timesteps, 1)
x4
x5 = np.append(x[-1].reshape(1,1,1), x4[-1,0:(timesteps-1),:].reshape(1,(timesteps-1),1),axis=1)
x5
x6 = np.append(y[0].reshape(1,1,1), x5[-1,0:(timesteps-1),:].reshape(1,(timesteps-1),1),axis=1)
x6
x7 = np.append(y[1].reshape(1,1,1), x6[-1,0:(timesteps-1),:].reshape(1,(timesteps-1),1),axis=1)
x7
y1 = x2[:,-1]
y1

yhat = numpy.zeros(shape=(3,))
yhat[0] = y[0]
yhat[1] = y[1]
yhat[2] = y[2]

model = Sequential()
model.add(LSTM(3, input_shape=(x4.shape[1], x4.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x4, y1, epochs=1000, batch_size=1, verbose=2)

model.predict(x5)
model.predict(x6)


##############################################################
### A model with exogeneous variables

timesteps = 3

steps = 10


### time series of variable
train = np.array([i for i in range(1,100)])
x = train
### test sample of 3 vars
test = np.array([i for i in range(100,110)])
y = test

### scale training sample in range -1 to 1
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(x.reshape(-1,1))
x = scaler.transform(x.reshape(-1,1))

### Z is the first esternal variable - shift it with 1 obs so that for each value we don't 
### us ethe lag but the corresponding value as a predictor
z = np.array([0,1]*100)[1:len(x)+1]
### test sample of z
yz = np.array([0,1]*100)[len(x)+1:len(x)+1+steps]
### s is a second predictor/feature, sz is the test variable
s = np.array([0,0.5,1]*100)[1:len(x)+1]
sz = np.array([0,0.5,1]*100)[len(x)+1:len(x)+1+steps]

### append the time series and the predictors
x = np.append(np.append(x.reshape(x.shape[0],1),
                        z.reshape(x.shape[0],1),axis = 1),
              s.reshape(x.shape[0],1),axis=1)
x
z
features = x.shape[1] - 1
### same for the test sample

yz = np.append(yz.reshape(y.shape[0],1),
               sz.reshape(y.shape[0],1),axis=1)
yz



### function to preprocess input and output
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	return df


### process the input
x1 = timeseries_to_supervised(x,timesteps)
x1
x1 = x1[timesteps:]
x1
x2 = np.array(x1)
x2
x3 = x2[:,:timesteps*(features+1)]
x3
x4 = x3.reshape(x3.shape[0], timesteps, features+1)
x4

### y1 is the output
y1 = x2[:,-(features+1)]
y1


model = Sequential()
model.add(LSTM(3, input_shape=(x4.shape[1], x4.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x4, y1, epochs=1000, batch_size=1, verbose=2)

# Stateteful version - slower and less precise
# model = Sequential()
# model.add(LSTM(3, batch_input_shape=(1, x4.shape[1], x4.shape[2]), stateful=True))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x4, y1, epochs=1000, batch_size=1, verbose=2, shuffle=False)
	

### create an empty matrix where the outcome will be stored
yhat = numpy.zeros(shape=(steps,))

### the first predicted value is based on the three last obs of the development sample

x_1 = np.append(x[-1].reshape(1,1,features+1), x4[-1,0:(timesteps-1),:].reshape(1,(timesteps-1),features+1),axis=1)
x_1
yhat[0] = model.predict(x_1)

### predict the next X steps
for i in range(1,steps):
    x_n = np.append(np.append(yhat[i-1],yz[i-1,:]).reshape(1,1,features+1), x_1[-1,0:(timesteps-1),:].reshape(1,(timesteps-1),features+1),axis=1)
    yhat[i] = model.predict(x_n)
    x_1 = x_n


yhat = scaler.inverse_transform(yhat.reshape(steps,1))

trainPredictPlot = numpy.zeros(shape=(len(train)+steps,1))
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[:len(train), :] = train.reshape(len(train),1)
trainPredictPlot[len(train):len(train)+steps, :] = test.reshape(len(test),1)

testPredictPlot = numpy.zeros(shape=(len(train)+steps,1))
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train):len(train)+steps, :] = yhat
yhat0=yhat


plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()



