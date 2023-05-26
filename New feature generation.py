# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:56:02 2022

@author: ZAK0131
"""


import pandas as pd
import numpy as np
print(pd.__version__)


import pickle as pkl
import pandas as pd


import os
os.chdir('//tedfil01/DataDropDEV/PythonPOC/')
#import dbUtils as db
import pyodbc
import dbUtils as db

import typing
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import numpy as np
import math
from math import sqrt

import datetime

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
import requests
import io

with open("//tedfil01/DataDropDEV/PythonPOC/CQG/usdadata.pkl", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv("//tedfil01/DataDropDEV/PythonPOC/Upload_CSVs/usdaData.csv")

df = pd.read_csv("//tedfil01/DataDropDEV/PythonPOC/Upload_CSVs/usdaData.csv")
df
df_copy =  pd.read_csv("//tedfil01/DataDropDEV/PythonPOC/Upload_CSVs/usdaData.csv")
date_time = pd.to_datetime(df.pop('dt'), format = '%Y-%m-%d')

#date related features
df_copy.dt = pd.to_datetime(df_copy.dt)
df_copy.dtypes 
df['month'] = pd.DatetimeIndex(df_copy['dt']).month
df['month']

df['week'] =(( (pd.DatetimeIndex(df_copy['dt']).month) * 4) + ((pd.DatetimeIndex(df_copy['dt']).day)/7 ))

df['week'] = df['week'].astype(int)
df['week']

df['day'] = (pd.DatetimeIndex(df_copy['dt']).day)
df['day']


#New weather features


#TODO: Redo the entire dataset. Fix Planted value columns and Date time stamps. 


api_key = 'B3A5D45D-0A9A-3FA5-869D-15861BD57EF4'
base_url_api_get = 'http://quickstats.nass.usda.gov/api/api_GET/?key=' + api_key + '&'

commodity_name ='CORN' #change based on desired commodity
state = 'NE'
year = '1950' #use GE so shouldn't matter
 
parameters =    'source_desc=SURVEY&sector_desc=CROPS&group_desc=FIELD CROPS&commodity_desc=' + commodity_name + '&statisticcat_desc=PROGRESS&' + \
                            'agg_level_desc=STATE&state_alpha=' + state + '&year__GE=' + year + '&format=CSV'
        
full_url = base_url_api_get + parameters
        
        
response = requests.get(full_url)
content = response.content
data = pd.read_csv(io.StringIO(content.decode('utf-8')))
 
 
data = data[['unit_desc', 'short_desc', 'year','week_ending', 'Value']]

data = data[data['unit_desc'] == 'PCT PLANTED']



#Get Weather data
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Stations
from datetime import datetime



NE_addresses = ['2727 W 2ND ST HASTINGS, NE 68901-4608', '4009 6TH AVE KEARNEY, NE 68845-2386', '818 FERDINAND PLZ SCOTTSBLUFF, NE 69361-4401', '1202 S COTTONWOOD ST NORTH PLATTE, NE 69101-6295', '120 W FAIRFIELD CLAY CENTER, NE 68933-1437', '100 CENTENNIAL MALL N LINCOLN, NE 68508-3803']

from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent = 'young_shark_227')
        
start = datetime(1979, 6, 3)
end = datetime(2022, 6, 10)
  
location = geolocator.geocode(NE_addresses[0])
Ne_loc = Point(lat = location.latitude, lon = location.longitude, alt = location.altitude)




from meteostat import Daily

temp = (40.9667, -98.15)
temp2= Point(temp[0],temp[1])
wdata = (Daily(temp2, start, end).fetch())

wdata = wdata[['tavg','tmin','tmax', 'prcp', 'snow']]
wdata = wdata.fillna(0)

 
stations = Stations()
Stations.cache_dir = 'Downloads'
stations = stations.nearby(41.1158, -98.0017)
stations = stations.inventory('daily')
station = stations.fetch(200)

station = station[station['region'] == 'NE']

Nebraska_weather = Daily('KAUH0', start, end)
Nebraska_weather = Nebraska_weather.fetch()
Nebraska_weather = Nebraska_weather.reset_index()

Nebby = Daily('KAUH0', start, end)
Nebby = Nebby.fetch()
print(Nebraska_weather)

Nebraska_weather = pd.DataFrame()

station = station.dropna()
station = station.reset_index()
for i in station.wmo.values:
    weather_temp = Daily(i, start, end)
    weather_temp = weather_temp.fetch()
    weather_temp = weather_temp.reset_index()
    weather_temp['station'] = i
    Nebraska_weather = Nebraska_weather.append(weather_temp, ignore_index = True)
    
#    Nebraska_weather = Nebraska_weather.concat({'station':i, 'date':weather_temp.time, 'tavg':weather_temp.tavg, 'tmin':weather_temp.tmin,
#                                               'tmax':weather_temp.tmax, 'prcp':weather_temp.prcp, 'snow':weather_temp.snow
#                                               }, ignore_index = True)

data = pd.read_csv('Downloads/planted_data_NE.csv')
Nebraska_weather = pd.read_csv('Downloads/Nebraska_weather.csv')


Nebraska_weather = Nebraska_weather[['time', 'tavg', 'tmin','tmax',
                                     'prcp', 'snow', 'station']]

data.dtypes
data.week_ending = pd.to_datetime(data.week_ending)

Nebraska_weather = Nebraska_weather[Nebraska_weather['station']]
station_groups = Nebraska_weather.groupby('station')

data2022 = data[data['week_ending'] >= '2022-04-02']
data2022 = data2022.reset_index( drop = True)
data.dtypes
percip_feature = pd.DataFrame(columns = ['date', 'station', 'num_percip'])

start = (data2022.week_ending.to_numpy())

end = (data2022.week_ending - timedelta(weeks=1)).to_numpy()
from datetime import date, timedelta

for name, group in station_groups:
    #print(group.time)
    for i in range(0, len(start)):
        print(str(name) + '_________________________')
        st = start[i]
        en = end[i]
        temp_query = group.query('time >= @en and time < @st')
        print(temp_query)
        Num_percip = (temp_query.prcp != 0).sum()
        percip_feature = percip_feature.append({'date':start[i], 'station':name, 'num_percip':Num_percip}, ignore_index = True)
        i = i+1




print(end)


test_query = Nebraska_weather.query('time >= @end and time < @start')

mask = (Nebraska_weather['time'] > end) & (Nebraska_weather['time'] <=start)
temp = Nebraska_weather.iloc(mask)



data.dtypes
data.week_ending = pd.to_datetime(data.week_ending)
Nebraska_weather.time = pd.to_datetime(Nebraska_weather.time)
Nebraska_weather.dtypes








##Days that it was cold feature: 
#Corn grows best in war, sunny weather. Count number of days the weather was below 45 degrees (7 degrees C)

Nebraska_weather.tavg = Nebraska_weather.tavg.fillna((Nebraska_weather.tmin + Nebraska_weather.tmax)/2)

station_groups = Nebraska_weather.groupby('station')



from datetime import date, timedelta
freezing_days_feature = pd.DataFrame(columns = ['date', 'station', 'num_cold'])

start = (data2022.week_ending.to_numpy())
end = (data2022.week_ending - timedelta(weeks=1)).to_numpy()

for name, group in station_groups:
    #print(group.time)
    for i in range(0, len(start)):
        print(name)
        st = start[i]
        en = end[i]
        temp_query = group.query('time >= @en and time < @st')
        print(temp_query)
        num_cold = (temp_query.tavg <= 7).sum()
        freezing_days_feature = freezing_days_feature.append({'date':start[i], 'station':name, 'num_cold':num_cold}, ignore_index = True)
        i = i+1


freezing_days_feature.to_csv('Downloads/freezing_days_feature.csv')

##Days it was optimal for corn growth - (warm weather ~70 F.  ) 

warm_days_feature = pd.DataFrame(columns = ['date', 'station', 'num_warm'])

start = (data2022.week_ending.to_numpy())
end = (data2022.week_ending - timedelta(weeks=1)).to_numpy()

for name, group in station_groups:
    #print(group.time)
    for i in range(0, len(start)):
        print(name)
        st = start[i]
        en = end[i]
        temp_query = group.query('time >= @en and time < @st')
        print(temp_query)
        num_warm = (temp_query.tavg >= 20).sum()
        warm_days_feature = warm_days_feature.append({'date':start[i], 'station':name, 'num_warm':num_warm}, ignore_index = True)
        i = i+1


warm_days_feature.to_csv('Downloads/warm_days_feature.csv')

##Absolute value of percipitation for past week.


Absolute_Percipitation = pd.DataFrame(columns = ['date', 'station', 'abs_percip'])

start = (data2022.week_ending.to_numpy())
end = (data2022.week_ending - timedelta(weeks=1)).to_numpy()

for name, group in station_groups:
    #print(group.time)
    for i in range(0, len(start)):
        print(name)
        st = start[i]
        en = end[i]
        temp_query = group.query('time >= @en and time < @st')
        print(temp_query)
        abs_percip = (temp_query.prcp).sum()
        Absolute_Percipitation = Absolute_Percipitation.append({'date':start[i], 'station':name, 'abs_percip':abs_percip}, ignore_index = True)
        i = i+1


Absolute_Percipitation.to_csv('Downloads/Absolute_Percipitation_feature.csv')


#NewDataAssembly:

Complete_Data_2022= pd.DataFrame()

abs_percip_groups = Absolute_Percipitation.groupby('station')
for name, group in abs_percip_groups:
    print(group.abs_percip)
    temp_col = group.abs_percip.reset_index()
    Complete_Data_2022['abs_percip_' + str(name) ] = temp_col.abs_percip
    

warm_days_group = warm_days_feature.groupby('station')
for name, group in warm_days_group:
    print(group.num_warm)
    temp_col = group.num_warm.reset_index()
    Complete_Data_2022['warm_days_' + str(name) ] = temp_col.num_warm

freezing_days_groups = freezing_days_feature.groupby('station')
for name, group in freezing_days_groups:
    print(group.num_cold)
    temp_col = group.num_cold.reset_index()
    Complete_Data_2022['cold_days_' + str(name) ] = temp_col.num_cold


num_percip_groups = percip_feature.groupby('station')
for name, group in num_percip_groups:
    print(group.num_percip)
    temp_col = group.num_percip.reset_index()
    Complete_Data_2022['num_percip_' + str(name) ] = temp_col.num_percip



Complete_Data_2022['Date'] = percip_feature['date']

Complete_Data_2022['Date']  = pd.to_datetime(Complete_Data_2022['Date'] )

Complete_Data_2022['month'] = pd.DatetimeIndex(Complete_Data_2022['Date']).month
Complete_Data_2022['month']

Complete_Data_2022['week'] =(( (pd.DatetimeIndex(Complete_Data_2022['Date']).month) * 4) + ((pd.DatetimeIndex(Complete_Data_2022['Date']).day)/7 ))

Complete_Data_2022['week'] = Complete_Data_2022['week'].astype(int)


Complete_Data_2022['day'] = (pd.DatetimeIndex(Complete_Data_2022['Date']).day)

Complete_Data_2022.dtypes


data = data.reset_index()

Complete_Data_2022['Percent_Planted'] = data2022['Value']

Complete_DataSet.to_csv('Downloads/Updated_Crop_Predict_DS.csv')

ds = pd.read_csv('Downloads/Updated_Crop_Predict_DS.csv')

DEV/PythonPOC/Upload_CSVs/Percent_Planted_Prediction.csv')

Complete_DataSet.to_csv('//tedfil01/DataDropDEV/PythonPOC/Upload_CSVs/Percent_Planted_Prediction.csv')
Complete_DataSet.to_csv('//tedfil01/DataDrop
                        
                        
Complete_data = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Upload_CSVs/Percent_Planted_Prediction.csv')




##AVG temp up
import statistics
    
usda.week_ending = pd.to_datetime(usda.week_ending)
    
start = (usda.week_ending.to_numpy())
end = (usda.week_ending - timedelta(weeks=1)).to_numpy()
station_groups = weather.groupby('station')

weather.tavg = weather.tavg.fillna((weather['tmin'] +weather['tmax'])/2)

avg_temp_upper = pd.DataFrame(columns = ['date', 'station', 'avg_temp_upper'])

for name, group in station_groups:
    for i in range(0, len(start)):
        print(str(name) + '_________________________')
        st = start[i]
        en = end[i]
        print(st)
        print(en)
        temp_query = group.query('time >= @en and time <= @st')
        print(temp_query)
        try:            
            avg_temp_up = statistics.mean(temp_query.tavg) + (2 * statistics.stdev(temp_query.tavg))
            print(statistics.mean(temp_query.tavg))
            print(statistics.stdev(temp_query.tavg))
            
        except:
            avg_temp_up= 0
        avg_temp_upper = avg_temp_upper.append({'date':start[i], 'station':name, 'avg_temp_upper':avg_temp_up}, ignore_index = True)
        i = i+1
    
temp_concat= pd.DataFrame()

avg_temp_upper__groups = avg_temp_upper.groupby('station')
for name, group in avg_temp_upper__groups:
    temp_col = group.avg_temp_upper.reset_index()
    temp_concat['avg_temp_upper' + str(name) ] = temp_col.avg_temp_upper







##AVG temp up
import statistics
    
usda.week_ending = pd.to_datetime(usda.week_ending)
    
start = (usda.week_ending.to_numpy())
end = (usda.week_ending - timedelta(weeks=1)).to_numpy()
station_groups = weather.groupby('station')

weather.tavg = weather.tavg.fillna((weather['tmin'] +weather['tmax'])/2)

avg_temp_lower = pd.DataFrame(columns = ['date', 'station', 'avg_temp_lower'])

for name, group in station_groups:
    for i in range(0, len(start)):
        print(str(name) + '_________________________')
        st = start[i]
        en = end[i]
        print(st)
        print(en)
        temp_query = group.query('time >= @en and time <= @st')
        print(temp_query)
        try:            
            avg_temp_low = statistics.mean(temp_query.tavg) - (2 * statistics.stdev(temp_query.tavg))
            print(statistics.mean(temp_query.tavg))
            print(statistics.stdev(temp_query.tavg))
            
        except:
            avg_temp_up= 0
        avg_temp_lower = avg_temp_lower.append({'date':start[i], 'station':name, 'avg_temp_lower':avg_temp_low}, ignore_index = True)
        i = i+1
    
temp_concat_lower = pd.DataFrame()

avg_temp_lower__groups = avg_temp_lower.groupby('station')

for name, group in avg_temp_lower__groups:
    temp_col = group.avg_temp_lower.reset_index()
    data['avg_temp_lower' + str(name) ] = temp_col.avg_temp_lower




statistics.stdev(weather.tavg[185013:185020])
statistics.mean(weather.tavg[185013:185020])

data = data.fillna(0)