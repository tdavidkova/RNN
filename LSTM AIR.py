# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:16:52 2020

@author: Teodora
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import datetime

#from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import warnings
warnings.filterwarnings("ignore")

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

df = pd.read_csv('/home/tdavidkova/Documents/Python/Data/airpassengers.csv')
df = pd.read_csv(r'home/tdavidkova/Documents/Python/Data\shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
df = pd.DataFrame(df)

df.Month = pd.to_datetime(df.Month)
df = df.set_index("Month")

n_test = 12
train, test = df[:-n_test], df[-n_test:]

scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

n_input = 12
n_features = 1
n_predict = 12
batch_size = 6

generator = TimeseriesGenerator(train, train, length=n_input, batch_size=batch_size)
len(generator)
#for i in range(len(generator)):
#	x, y = generator[i]
#	print('%s => %s' % (x, y))
    
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
#model.add(LSTM(200, batch_input_shape=(batch_size, n_input, n_features), stateful=True))

model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator,epochs=90)

pred_list = []

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_predict):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

#df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
#                          index=df[-n_test:-(n_test-n_predict)].index, columns=['Prediction'])
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=df[-n_input:].index, columns=['Prediction'])

df_test = pd.concat([df,df_predict], axis=1)

plt.figure(figsize=(20, 5))
plt.plot(df_test.index, df_test['Passengers'])
plt.plot(df_test.index, df_test['Prediction'], color='r')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()

type([2])
type(1)
[ train[-n_input:]]*6

[1,2,3]*6