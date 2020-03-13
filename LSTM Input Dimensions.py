#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:35:20 2020

@author: tdavidkova
"""

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
import numpy
from numpy import concatenate

x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([10,11,12])


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

x5 = np.append(x[-1].reshape(1,1,1), x4[-1,0:2,:].reshape(1,2,1),axis=1)
x5
x6 = np.append(y[0].reshape(1,1,1), x5[-1,0:2,:].reshape(1,2,1),axis=1)
x6
x7 = np.append(y[1].reshape(1,1,1), x6[-1,0:2,:].reshape(1,2,1),axis=1)
x7

yhat = numpy.zeros(shape=(3,))
yhat[0] = y[0]
yhat[1] = y[1]
yhat[2] = y[2]
