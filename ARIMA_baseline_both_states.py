# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 23:04:05 2022

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

def train_model(train):
    model = stats.ARIMA(train, order=best_pdq)  
    fitted = model.fit() 
    return fitted

def seperate_data(test_rescaled, pred_inverse_scale):
    cali_actual=[]
    org_actual=[]
    cali_pred=[]
    org_pred=[]

    for i in range(0, 30):
        if test_states[i] == 'California':
            cali_actual.append(test_rescaled[i])
            cali_pred.append(pred_inverse_scale[i])
        else:
            org_actual.append(test_rescaled[i])
            org_pred.append(pred_inverse_scale[i])
    
    return org_actual, org_pred, cali_actual, cali_pred

def transform(train_scaled, test_scaled):
    train_old = train_scaled.to_numpy()
    test_old = test_scaled.to_numpy()

    train_old = train_old.reshape(1368, )
    test_old = test_old.reshape(30, )

    train = pd.Series(train_old)
    test = pd.Series(test_old)
    return train, test

data = pd.read_csv('Oregon_cali_data.csv')
state = data['Province_State']
y = data['Confirmed']

# train-test split
n_train = 1368
train = y[:n_train]
test = y[n_train:]
test_states = state[n_train:].values.tolist()
#print(train.shape, test.shape)

train = train.to_numpy().reshape(-1, 1)
test = test.to_numpy().reshape(-1, 1)

# impute and normalize
train_imputed, test_imputed = median_im(train, test)
scaler, train_scaled, test_scaled = MinMax_Scaler(train_imputed, test_imputed)
#print(train_scaled.shape, test_scaled.shape)

train, test = transform(train_scaled, test_scaled)

# find the best combination of parameters for ARIMA
'''
p = range(0,8)
d = range(0,2)
q = range(0,8)

pdq_comb = list(itertools.product(p,d,q))

min_rmse_or = 100000000
min_rmse_ca = 100000000
best_pdq = (1,0,0)
org_ac = []
cali_ac = []
best_org_pred = []
best_cali_pred = []


for pdq in pdq_comb:
    print(pdq)
    model = stats.ARIMA(train, order=pdq)  
    fitted = model.fit()  
    pred = fitted.forecast(steps=30)
    
    pred_inverse_scale = scaler.inverse_transform(pred.to_numpy().reshape(30, 1))
    test_rescaled = scaler.inverse_transform(test.to_numpy().reshape(30, 1))
    
    cali_actual=[]
    org_actual=[]
    cali_pred=[]
    org_pred=[]

    for i in range(0, 30):
        if test_states[i] == 'California':
            cali_actual.append(test_rescaled[i])
            #print(i, 'cali_actual')
            cali_pred.append(pred_inverse_scale[i])
            #print(i, 'cali_pred')
        else:
            org_actual.append(test_rescaled[i])
            #print(i, 'org_actual')
            org_pred.append(pred_inverse_scale[i])
            #print(i, 'org_pred')
    org_ac = org_actual
    cali_ac = cali_actual
    # accuracy
    rmse_or = np.sqrt(metrics.mean_squared_error(org_actual, org_pred))
    rmse_ca = np.sqrt(metrics.mean_squared_error(cali_actual, cali_pred))
    
    if rmse_or < min_rmse_or and rmse_ca < min_rmse_ca:
        min_rmse_or = rmse_or
        min_rmse_ca = rmse_ca
        best_pdq = pdq
        best_org_pred = org_pred
        best_cali_pred = cali_pred

print("best RMSE OR", min_rmse_or)
print("best RMSE CA", min_rmse_ca)
print("best pdq", best_pdq)
'''
# best combination of (p,d,q) = (2, 1, 0) 
# model building, training
best_pdq = (2, 1, 0)
model = train_model(train)

# predict for 30 days
pred = model.forecast(steps=30)

# rescaling prediction and test data
pred_inverse_scale = scaler.inverse_transform(pred.to_numpy().reshape(30, 1))
test_rescaled = scaler.inverse_transform(test.to_numpy().reshape(30, 1))

# separate actual and predicted data by states to plot
org_actual, org_pred, cali_actual, cali_pred = seperate_data(test_rescaled, pred_inverse_scale)

# plotting results

# plot for Oregon
plt.plot(org_actual, marker='.', label="Actual Data")
plt.plot(org_pred, 'r', label="Prediction")
plt.ylabel('Confirmed Cases')
plt.xlabel('Period')
plt.legend()
plt.savefig('ARIMA_oregon_new.png')
plt.close()

# plot for California
plt.plot(cali_actual, marker='.', label="Actual Data")
plt.plot(cali_pred, 'r', label="Prediction")
plt.ylabel('Confirmed Cases')
plt.xlabel('Period')
plt.legend()
plt.savefig('ARIMA_california_new.png')
plt.close()

# accuracy for Oregon
print('RMSE for Oregon prediction', np.sqrt(metrics.mean_squared_error(org_actual, org_pred)))
# accuracy for California
print('RMSE for California prediction', np.sqrt(metrics.mean_squared_error(cali_actual, cali_pred)))

# accuracy --signed error
a = list(itertools.chain.from_iterable(org_actual))
b = list(itertools.chain.from_iterable(org_pred))
c = list(itertools.chain.from_iterable(cali_actual))
d = list(itertools.chain.from_iterable(cali_pred))

e = 0
for i in range(0, len(a)):
    e+= (a[i]-b[i])

f = 0
for i in range(0, len(c)):
    f+= (c[i]-d[i])
    
print('Signed error for OR', e)
print('Signed error for CA', f)
