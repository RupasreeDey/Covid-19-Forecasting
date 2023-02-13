# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 23:01:03 2022

@author: Rupasree Dey
AI539-Project
Winter Term-2022

Reference
https://towardsdatascience.com/lets-forecast-your-time-series-using-classical-approaches-f84eb982212c
https://machinelearningmastery.com/make-sample-forecasts-arima-python/
"""

# Import libraries
import itertools
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.impute import KNNImputer
from sklearn import *
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import statsmodels.tsa.arima.model as stats
from sklearn import metrics
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

def kNN(train, test):
    imputer = KNNImputer()
    imputer.fit(train)
    
    train_imputed = pd.DataFrame(imputer.transform(train))
    test_imputed = pd.DataFrame(imputer.transform(test))
    
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

def MaxAbs_Scaler(train, test):
    scaler = MaxAbsScaler()
    scaler.fit(train)
    # transform data
    scaled_train = pd.DataFrame(scaler.transform(train))
    scaled_test = pd.DataFrame(scaler.transform(test))
    
    return scaler, scaled_train, scaled_test

def MinMax_Scaler(train, test):
    scaler = MinMaxScaler()
    scaler.fit(train)
    # transform data
    scaled_train = pd.DataFrame(scaler.transform(train))
    scaled_test = pd.DataFrame(scaler.transform(test))
    
    return scaler, scaled_train, scaled_test

def Standard_Scaler(train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    # transform data
    scaled_train = pd.DataFrame(scaler.transform(train))
    scaled_test = pd.DataFrame(scaler.transform(test))
    
    return scaler, scaled_train, scaled_test

def inverse_scaling(test, pred):
    pred_inverse_scale = scaler.inverse_transform(pred.to_numpy().reshape(30, 1))
    pred_reshape = pred_inverse_scale.flatten()
    pred_final = pd.Series(pred_reshape)
    test_rescaled = scaler.inverse_transform(test.to_numpy().reshape(30, 1))
    return test_rescaled, pred_final

def transform(train_scaled, test_scaled):
    # convert list to numpy array
    train_old = train_scaled.to_numpy()
    test_old = test_scaled.to_numpy()

    train_old = train_old.reshape(669, )
    test_old = test_old.reshape(30, )

    # convert numpy array to series for ease in calculate accuacry
    train = pd.Series(train_old)
    test = pd.Series(test_old)
    return train, test

def train_model(train):
    model = stats.ARIMA(train, order=best_pdq)  
    fitted = model.fit() 
    return fitted

data = pd.read_csv('oregon_single.csv')

y = data['Confirmed']

# train-test split
train = y[:669]
test = y[669:]

train = train.to_numpy().reshape(-1, 1)
test = test.to_numpy().reshape(-1, 1)

# impute and scale
train_imputed, test_imputed = median_im(train, test)
scaler, train_scaled, test_scaled = MinMax_Scaler(train_imputed, test_imputed)

# reshape train and test data
train, test = transform(train_scaled, test_scaled)

# purpose of this portion is to find the best combination of ARIMA parameters p,q,d
'''
p = range(0,8)
d = range(0,2)
q = range(0,8)

pdq_comb = list(itertools.product(p,d,q))

min_rmse = 100000000
best_pred = pd.Series()
best_pdq = (1,1,1)

test_rescaled = scaler.inverse_transform(test.to_numpy().reshape(30, 1))

for pdq in pdq_comb:
    print(pdq)
    model = stats.ARIMA(train, order=pdq)  
    fitted = model.fit()  
    pred = fitted.forecast(steps=30)
    
    pred_inverse_scale = scaler.inverse_transform(pred.to_numpy().reshape(30, 1))
    pred_reshape = pred_inverse_scale.flatten()
    pred_final = pd.Series(pred_reshape)
    
    # accuracy
    rmse = np.sqrt(metrics.mean_squared_error(test_rescaled, pred_final))
    #print('RMSE', rmse)
    #print('MAE', metrics.mean_absolute_error(test_rescaled, pred_final))
    if rmse < min_rmse:
        min_rmse = rmse
        best_pdq = pdq
        best_pred = pred_final

print("best RMSE", min_rmse)
print("best pdq", best_pdq)
'''

#best pdq (2, 1, 7) with RMSE 1963.58
# ARIMA model building, training and generate predictions
best_pdq = (2, 1, 7)
model = train_model(train)  

# predict for 30 days
pred = model.forecast(steps=30)

# invserse scaling test and predicted data
test_rescaled, pred_final = inverse_scaling(test, pred)

# accuracy
rmse = np.sqrt(metrics.mean_squared_error(test_rescaled, pred_final))
print('RMSE', rmse)
# accuracy --signed error
test_list = list(test_rescaled.flatten())

signed_error = sum(test_list - pred_final)
print('Signed error', signed_error)

# plot
plt.plot(test_rescaled, marker='.', label="Actual Data")
plt.plot(pred_final, 'r', label="Prediction")
plt.ylabel('Confirmed Cases')
plt.xlabel('Period')
plt.legend()
plt.savefig('ARIMA_Oregon_single.png')
plt.close()