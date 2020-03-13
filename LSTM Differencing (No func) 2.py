#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:46:28 2020

@author: tdavidkova
"""

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
import numpy
from numpy import concatenate
 
# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('/home/tdavidkova/Documents/Python/Data/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

raw_values = series.values

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	return df


diff_values = difference(raw_values, 1)
diff_values = numpy.array(diff_values)

diff_values = diff_values.reshape(-1, 1)
train = diff_values[:-12,]
test = diff_values[-12:,]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train)
scaled = scaler.transform(train)

timesteps = 2

supervised = timeseries_to_supervised(scaled, timesteps)
supervised_values = supervised.values[timesteps:,:]
train_scaled = supervised_values

# train, test = supervised_values[0:-12, :], supervised_values[-12:, :]

# def scale(train, test):
# 	# fit scaler
# 	scaler = MinMaxScaler(feature_range=(-1, 1))
# 	scaler = scaler.fit(train)
# 	# transform train
# 	train = train.reshape(train.shape[0], train.shape[1])
# 	train_scaled = scaler.transform(train)
# 	# transform test
# 	test = test.reshape(test.shape[0], test.shape[1])
# 	test_scaled = scaler.transform(test)
# 	return scaler, train_scaled, test_scaled

# scaler, train_scaled, test_scaled = scale(train, test)

# def fit_lstm(train, batch_size, nb_epoch, neurons, timesteps):
X, y = train_scaled[:, 0:-1], train_scaled[:, -1]
X = X.reshape(X.shape[0], timesteps, 1)
X.shape,y.shape

neurons = 3
batch_size = 1

model = Sequential()
model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
# model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, y, epochs=10, batch_size=batch_size, verbose=2, shuffle=False)
model.reset_states()

Z = scaled[-timesteps:,]
Z = Z[::-1]
Z = Z.reshape(batch_size,  timesteps, 1)

# X, y = test_scaled[0:batch_size, 0:-1], test_scaled[0:batch_size, -1]
# X = X.reshape(batch_size, timesteps, 1)
# X.shape
yhat = model.predict(Z, batch_size=batch_size)
yhat_nom = scaler.inverse_transform(yhat)
yhat_nom = yhat_nom[0,0]
yhat_nom = train[-1]+yhat_nom

testA = numpy.zeros(shape=(12,1))
testA[:, :] = numpy.nan
testA[0,:] = yhat_nom

np.array(yhat).reshape(1,1,1)

Z =np.append(np.array(yhat).reshape(1,1,1), Z[:,:-1,:],axis=1)

yhat = model.predict(Z, batch_size=batch_size)
yhat_nom = scaler.inverse_transform(yhat)
yhat_nom = yhat_nom[0,0]
yhat_nom = testA[0,:]+yhat_nom

testA[1,:] = yhat_nom
# def invert_scale(scaler, X, yhat):
#  	new_row = [x for x in X] + [yhat]
#  	array = numpy.array(new_row)
#  	array = array.reshape(1, len(array))
#  	inverted = scaler.inverse_transform(array)
#  	return inverted[0, -1]
# yhat = invert_scale(scaler, train_scaled, yhat)



def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-1)
