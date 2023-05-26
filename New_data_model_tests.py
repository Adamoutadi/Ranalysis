# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:22:34 2022

@author: ZAK0131
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



Data = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Upload_CSVs/Percent_Planted_Prediction.csv')
Data = Data[Data['Date'] > '1979-06-03']
Data = Data.drop(columns = ['Unnamed: 0'])

Complete_Data_2022.to_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/2022_NE_Corn_fulldata.csv')
data2022 = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/2022_NE_Corn_fulldata.csv')


Data['Date'] = pd.to_datetime(Data.Date)
Data.index = Data.Date
Data = Data.drop(['Date'], axis = 1)

Data2022 = Complete_Data_2022
Data2022['Date'] = pd.to_datetime(Data2022.Date)
Data2022.index = Data2022.Date
Data2022 = Data2022.drop(['Date'], axis = 1)

model_errors = pd.DataFrame(columns = ['model', 'rmse', 'absolute_error'])


Nebraska_weather.station.value_counts()
##First: Random Forest Regression
from sklearn.model_selection import train_test_split
import copy
Data_copy = copy.deepcopy(Data)


Data_copy = Data_copy.drop(columns = ['abs_percip_72553','warm_days_72553', 'cold_days_72553', 'num_percip_72553'])

lel = lel.set_index('Date', inplace = True)
labels = np.array(features_L[11:298].Percent_Planted)
features = features_L[11:298].drop('Percent_Planted', axis = 1)
feature_list = list(features.columns)
features = np.array(features)

labels2022 = np.array(Data2022.Percent_Planted)
features2022 = Data2022.drop('Percent_Planted', axis =1)
features2022 = np.array(features2022)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42)


#print('Training Features Shape:', train_features.shape)
#print('Training Labels Shape:', train_labels.shape)
#print('Testing Features Shape:', test_features.shape)
#print('Testing Labels Shape:', test_labels.shape)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 61, random_state = 42)
regressor.fit(train_features, train_labels)

predictions = regressor.predict(features2022)

fig = plt.figure()
ax = plt.axes()
plt.xticks(rotation = 90)
ax.plot(Data2022.Date, labels2022, label = 'actual', marker = 'o')
ax.plot(Data2022.Date, predictions, label = 'predictions', marker = 'o')
ax.set_ylabel('percent planted 2022')
ax.set_xlabel('date')
plt.legend(loc = 'upper left')

fig.savefig('//tedfil01/DataDropDEV/PythonPOC/Amrith/2022_Planted_predictions.png')


errors = abs(predictions - labels2022)
print ("absolute error:" + str(np.mean(errors)))

rmseRF = np.sqrt(mean_squared_error(predictions,labels2022))
print("RMSE : % f" %(rmseRF))

#cross_validation_parameter tuning
errors_eval = pd.DataFrame(columns = ['n_estimator', 'abs_error', 'rmse'])
for i in range(10, 1000):
    regressor = RandomForestRegressor(n_estimators = i, random_state = 42)
    regressor.fit(train_features, train_labels)

    predictions = regressor.predict(test_features)
    errors = abs(predictions - test_labels)
    print ("absolute error:" + str(np.mean(errors)))

    rmseRF = np.sqrt(mean_squared_error(predictions,test_labels))
    print("RMSE : % f" %(rmseRF))
    errors_eval = errors_eval.append({'n_estimator':i, 'abs_error':errors, 'rmseRF':rmseRF}, ignore_index=True)
    i = i+10
min(errors_eval.rmseRF)    
#__Best num estimators is 61__
model_errors = model_errors.append({'model':'Random Forest', 'absolute_error':str(np.mean(errors)), 'rmse':rmseRF}, ignore_index=True)

from matplotlib.pyplot import figure
from sklearn.metrics import mean_squared_error

figure(figsize=(20, 20), dpi=80)
x = range(0, 87)
fig = plt.figure()
plt.plot(x, test_labels, color = 'blue' )
plt.plot(x, predictions, color = 'red')
plt.show()

plt.savefig('\\tepfil01\homes\ZAK0131\Downloads\image.png')

print(predictions)



##XGBoost

import xgboost as xg
from xgboost import XGBRegressor
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

DataXG = Data.values

X, Y = DataXG[:, :-1], DataXG[:, -1]

model = XGBRegressor()


model.fit(train_features, train_labels)

yhat= model.predict(test_features)

rmseXG = np.sqrt(mean_squared_error(yhat,test_labels))
print("RMSE : % f" %(rmseXG))


errors = abs(yhat - test_labels)
print ("absolute error:" + str(np.mean(errors)))

x = range(0, 87)
fig = plt.figure()
plt.plot(x, test_labels, color = 'blue' )
plt.plot(x, yhat, color = 'red')
plt.show()
from xgboost import plot_importance, plot_tree

_ = plot_importance(model, max_num_features = 15, height = 3)

model_errors = model_errors.append({'model':'XGBoost', 'absolute_error':str(np.mean(errors)), 'rmse':rmseXG}, ignore_index=True)

#cv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 1)

#Alternate implementation
#train_dmatrix = xg.DMatrix(data = train_features, label = train_labels)
#test_dmatrix = xg.DMatrix(data = test_features, label = test_labels)

#scores = cross_val_score(model, train_features, train_labels, scoring = 'neg_mean_absolute_error', cv = cv, n_jobs = 1)
#scores =absolute(scores)
#print(scores.mean())
#param = {"booster":"gbtree","verbosity":2, "validate_parameters": True}

#xgb_r = xg.train(params = param, dtrain = train_dmatrix, num_boost_round = 10)
#pred = xgb_r.predict(test_dmatrix)


#rmseXG = np.sqrt(mean_squared_error(pred,test_labels))
#print("RMSE : % f" %(rmseXG))


import sklearn
##Linear model:
lm= sklearn.linear_model.LinearRegression()
lm.fit(train_features, train_labels)

lin_tests = test_labels.reshape(-1,1 )
linear_preds = lm.predict(test_features)

import statsmodels.api as sm

X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())

rmseLin = np.sqrt(mean_squared_error(linear_preds,test_labels))
print("RMSE : % f" %(rmseLin))
errors = abs(linear_preds - test_labels)
print ("absolute error:" + str(np.mean(errors)))

model_errors = model_errors.append({'model':'Linear', 'absolute_error':str(np.mean(errors)), 'rmse':rmseLin}, ignore_index=True)


##RMSE(trained in usda_api_reload):

rmseLSTM = np.sqrt(mean_squared_error(predicted_y,test_y))
print("RMSE : % f" %(rmseLSTM))
errors = abs(predicted_y - test_y)
print ("absolute error:" + str(np.mean(errors)))

model_errors = model_errors.append({'model':'LSTM', 'absolute_error':str((np.mean(errors))*100), 'rmse':rmseLSTM*100}, ignore_index=True)
model_errors = model_errors.drop([2,3])



##ARIMA:



##ARIMA test:
#import statsmodels
'''
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot


Arima_data = data[['Date', 'Percent_Planted']]

X = Arima_data.Percent_Planted.values

size = int(len(X) * .75)
train,test = X[0:size], X[size:len(X)]

history = [x for x in train]
predictions = list()

for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, (predictions)))
print('Test RMSE: %.3f' % rmse)

pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

'''



























