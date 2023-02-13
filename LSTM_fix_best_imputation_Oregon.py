# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 22:05:08 2022

@author: Rupasree Dey
AI539-Project
Winter Term-2022

Reference
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error

def kNN(val, val1):
    imputer = KNNImputer().fit(val)
    train_imputed = imputer.transform(val)
    test_imputed = imputer.transform(val1)
    return train_imputed, test_imputed

def mean_im(train, test):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(train)
    
    train_imputed = imputer.transform(train)
    test_imputed = imputer.transform(test)
    
    return train_imputed, test_imputed

def median_im(train, test):
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer.fit(train)
    
    train_imputed = imputer.transform(train)
    test_imputed = imputer.transform(test)
    
    return train_imputed, test_imputed
 
# convert series to supervised learning
def series_to_supervised(data, n_in=14, n_out=1, dropnan=True):
    # n_in is number of past days to consider, n_out is number of future days to predict
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
df = read_csv('oregon_single.csv')

df = df.drop(['Province_State', 'Last_Update', 'Country_Region', 'Lat', 'Long',
       'Recovered', 'Active', 'FIPS',
       'UID', 'ISO3', 'Case_Fatality_Ratio'], axis=1)

# impute temporarily to convert into supervised learning problem
df.fillna(-1, inplace = True)
values = df.values

# ensure all data is float
values = values.astype('float32')

# convert to supervised learning
reframed = series_to_supervised(values, 1, 1)

# drop columns that don't come into prediction(drop all columns for t'th time except the target column)
reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
#print(reframed.head())

values = reframed.values

# train-test split
n_train = 668
train = values[:n_train, :]
test = values[n_train:, :]

df1 = pd.DataFrame(train)
df2 = df1.replace(-1, np.NaN)
val = df2.values

df3 = pd.DataFrame(test)
df4 = df3.replace(-1, np.NaN)
val1 = df4.values

# imputation
train_imputed, test_imputed = median_im(val, val1)

# split into input and outputs
train_X, train_y = train_imputed[:, :-1], train_imputed[:, -1]
test_X, test_y = test_imputed[:, :-1], test_imputed[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# build model
model = Sequential()
model.add(LSTM(256, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# train
history = model.fit(train_X, train_y, epochs=100, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=False)
 
# make prediction
yhat = model.predict(test_X)
text_X_in = test_X
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# rebuild data by concatenating predictions and test data
t = pd.DataFrame(test_X)
y = pd.DataFrame(yhat)
inv_yhat = pd.concat([t,y], axis=1)
inv_yhat_np = DataFrame.to_numpy(inv_yhat)

y1 = pd.DataFrame(test_y)
inv_y = pd.concat([t,y1], axis=1)
inv_y_np = DataFrame.to_numpy(inv_y)

y = inv_y_np[:, -1]
y_pred = inv_yhat_np[:, -1]

# calculate accuracy
rmse = np.sqrt(mean_squared_error(y, y_pred))
print('RMSE', rmse)

# accuracy --signed error
test_list = list(y.flatten())
pred_list = list(y_pred.flatten())
a = test_list
b = pred_list

e = 0
for i in range(0, len(a)):
    e+= (a[i]-b[i])
    
print('Signed error for single OR', e)

# plot actual and predicted for Oregon
plt.close()
plt.plot(y, marker='.', label="Actual Data")
plt.plot(y_pred, 'r', label="Prediction")
plt.ylabel('Confirmed Cases')
plt.xlabel('Period')
plt.legend()
plt.savefig('LSTM_challenge01.png')
plt.close()
