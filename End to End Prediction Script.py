# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:31:38 2022

@author: ZAK0131
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Stations
from datetime import datetime
from geopy.geocoders import Nominatim

import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import math
from math import sqrt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import requests
import io
from datetime import datetime, date, timedelta
from matplotlib.pyplot import figure
import joblib
import pickle
##PUlLING USDA DATA

def get_usda_data ( state, commodity_name, year):
    api_key = 'B3A5D45D-0A9A-3FA5-869D-15861BD57EF4'
    base_url_api_get = 'http://quickstats.nass.usda.gov/api/api_GET/?key=' + api_key + '&'

    commodity_name = commodity_name #change based on desired commodity
    state = state
     
    parameters =    'source_desc=SURVEY&sector_desc=CROPS&group_desc=FIELD CROPS&commodity_desc=' + commodity_name + '&statisticcat_desc=PROGRESS&' + \
                            'agg_level_desc=STATE&state_alpha=' + state + '&year__GE=' + year + '&format=CSV'
        
    full_url = base_url_api_get + parameters
        
    response = requests.get(full_url)
    content = response.content
    data = pd.read_csv(io.StringIO(content.decode('utf-8')))
 
    data = data[['unit_desc', 'short_desc', 'year','week_ending', 'Value']]
    data = data[data['unit_desc'] == 'PCT PLANTED']
    
    return data

#usda = get_usda_data('NE', 'CORN', '1980')

#input latitude and Longitude of central location of state
def get_weather_data (latitude, longitude, state):
    start = datetime(1979, 6, 3)
    end = datetime(2022, 6, 10)
    
    stations = Stations()
    Stations.cache_dir = 'Downloads'
    stations = stations.nearby(latitude, longitude)
    stations = stations.inventory('daily')
    station = stations.fetch(200)
    
    station = station[station['region'] == state]

    state_weather = pd.DataFrame()

    #station = station.dropna()
    #station = station.reset_index()
    for i in station.index.values:
        weather_temp = Daily(i, start, end)
        weather_temp = weather_temp.fetch()
        weather_temp = weather_temp.reset_index()
        weather_temp['station'] = i
        state_weather = state_weather.append(weather_temp, ignore_index = True)
    
    return state_weather,station

#weather = get_weather_data(41.1158, -98.0017, 'NE')

import statistics
def feature_gen (weather_data, usda_data):
    percip_feature = pd.DataFrame(columns = ['date', 'station', 'num_percip'])
    freezing_days_feature = pd.DataFrame(columns = ['date', 'station', 'num_cold'])
    warm_days_feature = pd.DataFrame(columns = ['date', 'station', 'num_warm'])
    Absolute_Percipitation = pd.DataFrame(columns = ['date', 'station', 'abs_percip'])
    avg_temp_upper = pd.DataFrame(columns = ['date', 'station', 'avg_temp_upper'])
    avg_temp_lower = pd.DataFrame(columns = ['date', 'station', 'avg_temp_lower'])

    weather_data.tavg = weather_data.tavg.fillna((weather_data['tmin'] +weather_data['tmax'])/2)

    usda_data.week_ending = pd.to_datetime(usda_data.week_ending)
    
    start = (usda_data.week_ending.to_numpy())
    end = (usda_data.week_ending - timedelta(weeks=1)).to_numpy()
    station_groups = weather_data.groupby('station')
    
    for name, group in station_groups:
    #print(group.time)
        for i in range(0, len(start)):
            print(str(name) + '_________________________')
            st = start[i]
            en = end[i]
            temp_query = group.query('time >= @en and time <= @st')
            #print(temp_query)
            Num_percip = (temp_query.prcp != 0).sum()
            percip_feature = percip_feature.append({'date':start[i], 'station':name, 'num_percip':Num_percip}, ignore_index = True)
            num_cold = (temp_query.tavg <= 7).sum()
            freezing_days_feature = freezing_days_feature.append({'date':start[i], 'station':name, 'num_cold':num_cold}, ignore_index = True)
            num_warm = (temp_query.tavg >= 20).sum()
            warm_days_feature = warm_days_feature.append({'date':start[i], 'station':name, 'num_warm':num_warm}, ignore_index = True)
            abs_percip = (temp_query.prcp).sum()
            Absolute_Percipitation = Absolute_Percipitation.append({'date':start[i], 'station':name, 'abs_percip':abs_percip}, ignore_index = True)
            try:            
                avg_temp_up = statistics.mean(temp_query.tavg) + (2 * statistics.stdev(temp_query.tavg))
            except:
                avg_temp_up= 0
            avg_temp_upper = avg_temp_upper.append({'date':start[i], 'station':name, 'avg_temp_upper':avg_temp_up}, ignore_index = True)
            try:            
                avg_temp_low = statistics.mean(temp_query.tavg) - (2 * statistics.stdev(temp_query.tavg))
            except:
                avg_temp_low= 0
            avg_temp_lower = avg_temp_lower.append({'date':start[i], 'station':name, 'avg_temp_lower':avg_temp_low}, ignore_index = True)
    
            i = i+1
    
    Complete_Data= pd.DataFrame()

    abs_percip_groups = Absolute_Percipitation.groupby('station')
    for name, group in abs_percip_groups:
        #print(group.abs_percip)
        temp_col = group.abs_percip.reset_index()
        Complete_Data['abs_percip_' + str(name) ] = temp_col.abs_percip
        
    
    warm_days_group = warm_days_feature.groupby('station')
    for name, group in warm_days_group:
        #print(group.num_warm)
        temp_col = group.num_warm.reset_index()
        Complete_Data['warm_days_' + str(name) ] = temp_col.num_warm
    
    freezing_days_groups = freezing_days_feature.groupby('station')
    for name, group in freezing_days_groups:
        #print(group.num_cold)
        temp_col = group.num_cold.reset_index()
        Complete_Data['cold_days_' + str(name) ] = temp_col.num_cold
    
    
    num_percip_groups = percip_feature.groupby('station')
    for name, group in num_percip_groups:
        #print(group.num_percip)
        temp_col = group.num_percip.reset_index()
        Complete_Data['num_percip_' + str(name) ] = temp_col.num_percip
        
    avg_temp_upper__groups = avg_temp_upper.groupby('station')
    for name, group in avg_temp_upper__groups:
        temp_col = group.avg_temp_upper.reset_index()
        Complete_Data['avg_temp_upper' + str(name) ] = temp_col.avg_temp_upper
     
    avg_temp_lower__groups = avg_temp_lower.groupby('station')
    for name, group in avg_temp_lower__groups:
        temp_col = group.avg_temp_lower.reset_index()
        Complete_Data['avg_temp_lower' + str(name) ] = temp_col.avg_temp_lower
        
    Complete_Data = Complete_Data.fillna(0)
    return Complete_Data


#data = feature_gen(weather, usda)


def date_features (data, usda):
    usda = usda.reset_index()
    data['Date'] = usda['week_ending']
    data['Percent_Planted'] = usda['Value']   
    data['Date']  = pd.to_datetime(data['Date'] )
    data['month'] = pd.DatetimeIndex(data['Date']).month
    data['week'] =(( (pd.DatetimeIndex(data['Date']).month) * 4) + ((pd.DatetimeIndex(data['Date']).day)/7 ))
    data['week'] = data['week'].astype(int)
    data['day'] = (pd.DatetimeIndex(data['Date']).day)
    
    return data
    
#data = date_features(data, usda)

def model (data, train_start_date, test_start_date):  
    print(data.columns)
    data['Date'] = pd.to_datetime(data.Date)
    data = data.reset_index(drop = True)
    
    labels_train = data[(data['Date']>=train_start_date) & (data['Date'] < test_start_date) ]
    train_dates = labels_train.Date
    labels_train = labels_train.drop(['Date'], axis = 1)
    labels_train = np.array(labels_train.Percent_Planted)
    
    labels_test = data[data['Date'] >= test_start_date ]
    test_dates = labels_test.Date
    labels_test = labels_test.drop(['Date'], axis = 1)
    labels_test = np.array(labels_test.Percent_Planted)
    
    features_train =  data[(data['Date']>=train_start_date) & (data['Date'] < test_start_date)]
    features_train = features_train.drop(['Date'], axis = 1)
    features_train = np.array(features_train.drop('Percent_Planted', axis = 1))
   
    features_test =  data[data['Date'] >= test_start_date ]
    features_test = features_test.drop(['Date'], axis = 1)
    features_test =  np.array(features_test.drop('Percent_Planted', axis = 1))
    
    regressor = RandomForestRegressor(n_estimators = 80, random_state = 42)
    regressor.fit(features_train, labels_train)

    predictions = regressor.predict(features_test)
    
    abs_error = abs(predictions - labels_test)
    abs_error = (np.mean(abs_error))
    print ("absolute error:" + str(abs_error))

    rmse = np.sqrt(mean_squared_error(predictions,labels_test))
    print("RMSE : % f" %(rmse))
    
   # weights = regressor.save()
    
    return predictions, abs_error, rmse, labels_test, test_dates, train_dates, regressor
    
#preds,abs_error,rmse,labels_test, test_dates, train_dates = model(data, '1980-04-27', '2018-04-08')


    
def visualizations (data, predictions, actual_values, state, commodity, test_dates):
    
    fig = plt.figure(figsize = (12,6), dpi = 80)
    ax = plt.axes()
    plt.xticks(rotation = 90)
    ax.plot(test_dates[0:11], actual_values[0:11], label = 'actual', marker = 'o')
    ax.plot(test_dates[0:11], predictions[0:11], label = 'predictions', marker = 'o')
    ax.set_ylabel('percent ' + commodity + ' planted' + ' in ' + state)
    ax.set_xlabel('date')
    plt.legend(loc = 'upper left')
    plt.savefig('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/'+commodity + '_' + state + 'plot.png')
    plt.show()

#visualizations(data, preds, labels_test, 'NE', 'CORN')


def main(state, commodity, start_year, central_lat, central_lon, model_train_start, model_test_start):
    
    usda = get_usda_data(state, commodity, start_year)
    weather = get_weather_data(central_lat, central_lon, state)
    data = feature_gen(weather, usda)
    data = date_features(data, usda)
    preds,abs_error,rmse,labels_test, test_dates, train_dates = model(data,model_train_start , model_test_start)
    visualizations(data, preds, labels_test, state, commodity, test_dates)
    
#main('NE', 'CORN', '1980', 38.5489, -89.1270, '1980-04-27' , '2018-04-08')


#Lat Long combos: 
#NE: 41.1158, -98.001
#IA: 42.4903, -94.2040 --> RMSE: 13
#IL: 38.5489, -89.1270 --> RMSE: 23.497 
#MN: 44.9765, -93.2761
#IN: 38.0998, -86.1586
#KS: 38.7306, -98.2281
#ND: 47.1164, -101.2996
    
#CO: 39.803318, -105.516830
#KY: 38.047989, -84.501640
#MI: 44.182205, -84.506836
#MO: 38.573936, -92.603760
#NC: 35.787743, -78.644257
#OH: 40.367474, -82.996216 
#PA 40.335648, -75.926872
#SD: 43.969515, -99.901813
#TN: 35.860119, -86.660156
#TX:  31.9685988, -99.9018131
#WI: 43.470364, -88.862572


#AR: 34.154999, -93.073425
#CA:  36.778259 , -119.417931
#ID: 44.068202,-114.742041
#MT: 46.965260, -109.533691
#OK: 35.481918, -97.508469
#OR: 44.000000, -120.500000
#WA: 47.608013, -122.335167
#LA 30.984298 and -91.962333 
#States currently not dog shit for CORN: NE(RMSE: 8.09), KS(RMSE:9.5)
#States currently not dog shit for WHEAT: NE(RMSE: 6.8), KS(RMSE:7)


##################################
usda = get_usda_data('IL', 'SOYBEANS', '1980')
weather,station = get_weather_data(38.5489, -89.1270,  'IL')
data = feature_gen(weather, usda)
data = date_features(data, usda)
data = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/NE_Wheat_fulldata.csv')
preds,abs_error,rmse,labels_test, test_dates, train_dates, regressor = model(data,'1980-01-27' , '2020-04-08')
visualizations(data, preds, labels_test, 'NE', 'WHEAT', test_dates)

#########################

d = {'Commodity': ['WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT',
                   'WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT',
                   'WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT'],
     'State': ['NE', 'IA', 'IL', 'MN', 'IN', 'KS', 'ND', 
               'CO', 'KY', 'MI', 'MO', 'NC', 'OH', 'PA', 'SD', 'TN', 'TX', 'WI', 'AR', 'CA', 'ID', 'MT', 'OK', 'OR', 'WA'],
     'lat': [ 41.1158, 42.4903, 38.5489, 44.9765,  38.0998,  38.7306, 47.1164,
             39.803318,  38.047989,  44.182205, 38.573936, 35.787743, 40.367474,  40.335648,43.969515,35.860119,31.9685988,43.470364,
             34.154999, 36.77825,44.068202, 46.965260,35.481918,44.000000,47.608013],
     'lon':[ -98.001, -94.2040, -89.1270, -93.2761, -86.1586, -98.2281, -101.2996,
            -105.516830, -84.501640, -84.506836,-92.603760, -78.644257, -82.996216 ,-75.926872, 99.901813,-86.660156,-99.9018131,-88.862572,
             -93.073425,  -119.417931, -114.742041, -109.533691, -97.508469, -120.500000, -122.335167]}
     
     
d_wheat = {'Commodity': ['WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT',
                   'WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT','WHEAT',
                   'WHEAT','WHEAT','WHEAT','WHEAT','WHEAT'],
     'State': ['IL', 'IN', 'KS', 'ND', 
               'CO', 'KY', 'MI', 'MO', 'NC', 'OH', 'PA', 'SD', 'TN', 'TX', 'WI', 'AR', 'CA', 'ID', 'MT', 'OK', 'OR', 'WA'],
     'lat': [ 38.5489,  38.0998,  38.7306, 47.1164,
             39.803318,  38.047989,  44.182205, 38.573936, 35.787743, 40.367474,  40.335648,43.969515,35.860119,31.9685988,43.470364,
             34.154999, 36.77825,44.068202, 46.965260,35.481918,44.000000,47.608013],
     'lon':[ -89.1270,  -86.1586, -98.2281, -101.2996,
            -105.516830, -84.501640, -84.506836,-92.603760, -78.644257, -82.996216 ,-75.926872, 99.901813,-86.660156,-99.9018131,-88.862572,
             -93.073425,  -119.417931, -114.742041, -109.533691, -97.508469, -120.500000, -122.335167]}

d_soybean = {'Commodity': ['SOYBEANS','SOYBEANS','SOYBEANS','SOYBEANS','SOYBEANS','SOYBEANS','SOYBEANS','SOYBEANS','SOYBEANS','SOYBEANS'
                           ,'SOYBEANS','SOYBEANS','SOYBEANS','SOYBEANS','SOYBEANS','SOYBEANS','SOYBEANS','SOYBEANS','SOYBEANS'],
     'State': ['LA','NE', 'IA', 'IL', 'MN', 'IN', 'KS', 'ND', 
               'CO', 'KY', 'MI', 'MO', 'NC', 'OH', 'PA', 'SD', 
               'TN',  'WI', 'AR'],
     'lat': [ 30.984298, 41.1158, 42.4903, 38.5489, 44.9765,  38.0998,  38.7306, 47.1164,
             39.803318,  38.047989,  44.182205, 38.573936, 35.787743, 40.367474,  40.335648,43.969515,35.860119,43.470364,
             34.154999],
     'lon':[ -91.962333, -98.001, -94.2040, -89.1270, -93.2761, -86.1586, -98.2281, -101.2996,
            -105.516830, -84.501640, -84.506836,-92.603760, -78.644257, -82.996216 ,-75.926872, 99.901813,-86.660156,-88.862572,
             -93.073425]}
     
tests = pd.DataFrame(data = d_soybean)

results = pd.DataFrame(columns = ['State', 'Commodity', 'RMSE', 'ABS'])

#tests[:]

tests = tests[16:]
tests = tests.reset_index()

for i in tests.index.values:
    temp = tests.iloc[[i]]
    temp = temp.reset_index()
    state = temp.State
    commodity = temp.Commodity
    lat = temp.lat
    lon = temp.lon
    print(temp)
    print()
    usda = get_usda_data(state.iat[0], commodity.iat[0], '1980')
    weather,station = get_weather_data(lat.iat[0], lon.iat[0], state.iat[0])
    data = feature_gen(weather, usda)
    data = date_features(data, usda)
    data.to_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/'+state.iat[0]+'_soybeans_fulldata.csv')
    preds,abs_error,rmse,labels_test, test_dates, train_dates, regressor = model(data,'1980-04-27' , '2018-04-08')
    #visualizations(data, preds, labels_test, 'IN', 'WHEAT', test_dates)
    results = results.append({'State':state, 'Commodity':commodity, 'RMSE':rmse, 'ABS':abs_error}, ignore_index = True)
    


results.to_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/corn_results.csv')
usda = get_usda_data('IA', 'WHEAT', '1980')

for i in d['State']:
    data = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/'+i+'_Corn_fulldata.csv')
    preds,abs_error,rmse,labels_test, test_dates, train_dates, regressor = model(data,'1980-04-27' , '2018-04-08')
    results = results.append({'State':i, 'Commodity':'CORN', 'RMSE':rmse, 'ABS':abs_error}, ignore_index = True)



'''

data = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/IL_Corn_fulldata.csv')

data.Date  = pd.to_datetime(data.Date)


test_IL = data[data.Date > '2015-01-01']
test_IL_labels = test_IL.Percent_Planted
test_IL_feats = test_IL.drop('Percent_Planted', axis = 1)
test_IL_feats = test_IL_feats.drop('Date', axis = 1)

Compiled_df.Date = 0
import copy
Data_copy = copy.deepcopy(df)
Data_copy = Data_copy[((Data_copy.Date <'2019-03-31') | (Data_copy.Date >'2019-06-30'))]
Data_copy = Data_copy[Data_copy.Date >= '2006-04-10']

IL_generated_data[IL_generated_data <0] = 0
labels = np.array(IL_generated_data.Percent_Planted)
features = IL_generated_data.drop('Percent_Planted', axis = 1)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42)
    
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 42)
regressor.fit(train_features, train_labels)

predictions = regressor.predict(test_features)
predictions = regressor.predict(np.array(test_IL_feats))


abs_error = abs(preds - test_IL_labels[0:32])
abs_error = (np.mean(abs_error))
print ("absolute error:" + str(abs_error))
rmse = np.sqrt(mean_squared_error(predictions,  test_labels))
print("RMSE : % f" %(rmse))
    
features = features.drop(columns = ['Date'])

date_test_NE = dataNE.drop('Percent_Planted', axis =1)