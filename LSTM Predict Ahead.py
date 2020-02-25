#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:52:38 2020

@author: tdavidkova
"""

import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt
import math
dataframe = pd.read_csv('/home/tdavidkova/Documents/Python/Data/airpassengers.csv', usecols=[1], engine='python')
# dataframe = pd.read_csv('/home/tdavidkova/Projects/VMF/Data/bshmills_sales.csv', usecols=[4], engine='python')

plt.plot(dataframe)
plt.show()

# number of lags used
look_back = 36

# number of data points to predict
steps = 12

	
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

	
# fix random seed for reproducibility
numpy.random.seed(7)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# n = math.ceil(len(dataset)/12)

# month = numpy.array([i for k in range(n) for i in range(12)])
# month.shape
	
# split into train and test sets
test_size = 12
train_size = int(len(dataset) - test_size)

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t and Y=t+1

trainX, trainY = create_dataset(train, look_back)

# month_train = numpy.reshape(month[:trainX.shape[0]],(trainX.shape[0],1))
# trainX = numpy.append(trainX, month_train, axis=1)

	

# reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

	
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(5, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


testX = train[-look_back:]


# add variable
# month_test = numpy.reshape(month[trainX.shape[0]],(1,1))
# testX = numpy.append(testX, month_test, axis=0)
    
# testX = numpy.reshape(testX, (testX.sha   pe[1], 1, testX.shape[0]))
testX = numpy.reshape(testX, (testX.shape[1], testX.shape[0], 1))
testX.shape
# make predictions


testPredict = model.predict(testX)
testA = numpy.zeros(shape=(steps,1))
testA[:, :] = numpy.nan
testA[0,:] = testPredict
testA.shape
testA
trainX.shape

# simplest way
# for i in range(steps-1):
#     X = [testX[0,0,k] for k in range(1,look_back)]
#     X.append(testPredict[0,0])
#     testX = numpy.array(X,ndmin=3)
#     testPredict = model.predict(testX)
#     testA[i+1,:] = testPredict
    
# add variable number of months
# for i in range(steps-1):
#     X = [testX[0,0,k] for k in range(1,look_back)]
#     X.append(testPredict[0,0])
#     month_test = month[trainX.shape[0]+i+1]
#     X.append(month_test)
#     testX = numpy.array(X,ndmin=3)
#     testPredict = model.predict(testX)
#     testA[i+1,:] = testPredict    
    
# add time steps variable    
for i in range(steps-1):
    X = [testX[0,k,0] for k in range(1,look_back)]
    X.append(testPredict[0,0])
    testX = numpy.array(X,ndmin=3)
    testX = numpy.reshape(testX, (1, look_back, 1))
    testPredict = model.predict(testX)
    testA[i+1,:] = testPredict    

testA = scaler.inverse_transform(testA)
test = scaler.inverse_transform(test)
testA
test

trainPredictPlot = numpy.zeros(shape=(len(train)+steps,1))
trainPredictPlot[:, :] = numpy.nan
train = scaler.inverse_transform(train)
trainPredictPlot[:len(train), :] = train
trainPredictPlot[len(train):len(train)+len(test), :] = test

testPredictPlot = numpy.zeros(shape=(len(train)+steps,1))
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train):len(train)+len(testA), :] = testA

plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


testA
