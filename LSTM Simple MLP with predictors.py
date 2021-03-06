#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:52:33 2020

@author: tdavidkova
"""


import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

timesteps = 5

steps = 10

train = np.array([i for i in range(1,100)])
x = train
### test sample of 3 vars
test = np.array([i for i in range(100,110)])
y = test

### scale training sample in range -1 to 1
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(x.reshape(-1,1))
x = scaler.transform(x.reshape(-1,1))
y = scaler.transform(y.reshape(-1,1))

### Z is the first esternal variable 

z = np.array([0,1]*100)[:len(x)]
### test sample of z
yz = np.array([0,1]*100)[len(x):len(x)+steps]
### s is a second predictor/feature, sz is the test variable
s = np.array([0,0.5,1]*100)[:len(x)]
ys = np.array([0,0.5,1]*100)[len(x):len(x)+steps]



# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

x1, y1 = create_dataset(x,timesteps)
x1

x2 = np.append(x1,z[timesteps:].reshape(-1,1),axis=1)
x2
x3 = np.append(x2,s[timesteps:].reshape(-1,1),axis=1)
x3
x4 = np.reshape(x3, (x3.shape[0], 1, x3.shape[1]))
x4

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(3, input_shape=(x4.shape[1], x4.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x4, y1, epochs=1000, batch_size=1, verbose=2)


x_1 = np.append(np.append(x[-timesteps:].reshape(1,-1), yz[0]),ys[0]).reshape(1,1,x4.shape[2])
x_1
yhat = np.zeros(shape=(steps,))
yhat[0] = model.predict(x_1)


# x_n = np.append(np.append(np.append(x_1[:,:,1:timesteps].reshape(1,-1),yhat[i-1]),yz[i]),ys[i]).reshape(1,1,4)
   
### predict the next X steps
for i in range(1,steps):
    x_n = np.append(np.append(np.append(x_1[:,:,1:timesteps].reshape(1,-1),yhat[i-1]),yz[i]),ys[i]).reshape(1,1,x4.shape[2])
    yhat[i] = model.predict(x_n)
    x_1 = x_n
    
yhat



yhat = scaler.inverse_transform(yhat.reshape(steps,1))

trainPredictPlot = np.zeros(shape=(len(train)+steps,1))
trainPredictPlot[:, :] = np.nan
trainPredictPlot[:len(train), :] = train.reshape(len(train),1)
trainPredictPlot[len(train):len(train)+steps, :] = test.reshape(len(test),1)

testPredictPlot = np.zeros(shape=(len(train)+steps,1))
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train):len(train)+steps, :] = yhat


plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
