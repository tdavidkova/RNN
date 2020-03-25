#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:35:20 2020

@author: tdavidkova
"""
#### Build NN LSTM model for prediction of X outcomes ahead


import numpy as np
import pandas as pd
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
from keras.utils import to_categorical

# fix random seed for reproducibility
numpy.random.seed(7)

#### Model without exogeneous variables, see below general case - exogeneous variables

x = np.array([i for i in range(1,37)])
y = np.array([i for i in range(37,40)])

dataframe = pd.read_csv('/home/tdavidkova/Documents/Python/Data/airpassengers.csv', usecols=[1], engine='python')



##############################################################
### A model with exogeneous variables

timesteps = 12

steps = 12


### time series of variable
train = dataframe.values[range(132)]
x = train
### test sample of 3 vars
test = dataframe.values[range(132,144)]
y = test

ex = np.array([i for i in range(12)]*15)
print(ex)
# one hot encode
ex = to_categorical(ex)
print(ex)

### scale training sample in range -1 to 1
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(x.reshape(-1,1))
x = scaler.transform(x.reshape(-1,1))
x



zs = ex[1:len(x)+1,:-1]

zsy = ex[len(x)+1:len(x)+1+steps,:-1]
### append the time series and the predictors
x = np.append(x, zs,axis = 1)
x

features = x.shape[1] - 1
### same for the test sample


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
model.fit(x4, y1, epochs=100, batch_size=1, verbose=2)

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
    x_n = np.append(np.append(yhat[i-1],zsy[i-1,:]).reshape(1,1,features+1), x_1[-1,0:(timesteps-1),:].reshape(1,(timesteps-1),features+1),axis=1)
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
# yhat0=yhat


plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()



