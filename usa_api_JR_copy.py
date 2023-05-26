# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:45:07 2022

@author: ZJM0667
"""

##THINGS TO DO:
#CONNECT PLANTED AND WEATHER
#RE-CALIBRATE ML MODEL/MAKE SURE IT WORKS - ADD IN INDEX FOR DATE TO WHERE IT IS 0,1,2,3,... THEN ITERATIVE FOR THE NEXT YEAR
#FIX WEATHER SCRAPPING TO ONLY INCLUDE A FEW OF THE USDA SITES AND THE EXTRA (LIKE WHAT DID FOR THE FIRST FEW STATES)


import pandas as pd
import requests
import csv
import urllib.request

from meteostat import Point, Daily
from meteostat import Stations

import os
os.chdir('//tedfil01/DataDropDEV/PythonPOC/CQG')

import io
from datetime import datetime

import matplotlib.pyplot as plt

from geopy.geocoders import Nominatim

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import math
from math import sqrt

import datetime

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import os
os.chdir('//tedfil01/DataDropDEV/PythonPOC/')
import dbUtils as db

from datetime import datetime
from datetime import timedelta

#import meteostat as mt



#rom c_usda_quick_stats import c_usda

#csv_file = requests.get('http://quickstats.nass.usda.gov/api/api_GET/?key=369213FD-5245-32C7-AD6F-E1F51EAA1DEA&commodity_desc=CORN&year__GE=2012&state_alpha=VA&format=CSV')

#df = pd.read_csv(csv_file)

api_key = '369213FD-5245-32C7-AD6F-E1F51EAA1DEA'
base_url_api_get = 'http://quickstats.nass.usda.gov/api/api_GET/?key=' + api_key + '&'


#states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
 #          'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
  #         'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
   #        'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
    #       'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

states = ['IA', 'IL','NE','MN','IN','KS','SD','OH', 'ND', 'WI','MO'] #relevant states

for a in states:
    
    
    
    
    
    if a == 'NE':
        
        print('works already')
        continue   
        
        commodity_name ='CORN' #change based on desired commodity
        state = 'NE'
        year = '1900' #use GE so shouldn't matter
        
        def getting_data(commodity_name, state, year='1900'):
            
            parameters =    'source_desc=SURVEY&sector_desc=CROPS&group_desc=FIELD CROPS&commodity_desc=' + commodity_name + '&statisticcat_desc=PROGRESS&' + \
                            'agg_level_desc=STATE&state_alpha=' + state + '&year__GE=' + year + '&format=CSV'
        
            full_url = base_url_api_get + parameters
        
        
            response = requests.get(full_url)
            
            content = response.content
            data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
            
            relevant_columns = ['year','week_ending','state_alpha','state_name','short_desc','Value'] # for corn at least
        
            for i in list(data.columns):
                if i not in relevant_columns:
                    data.drop(i, axis =1, inplace= True)
            
            data.set_index('week_ending', inplace = True)
            data.sort_index(ascending= True, inplace = True)
            
            global planted_values_dates
            
            planted_values = []
            planted_values_dates = []
            #print(data.index)
            for i in list(data.index):
                #print('PLANTED' in str(data.loc[i,'short_desc']))
                if 'PLANTED' in str(data.loc[i, 'short_desc']):
                    planted_values_dates.append(i) #need to figure out what to do about the series
                    #if type(data.loc[i,'Value']) == pd.Series:
                     #   print(data.loc[i,'Value'][1])
                        
                    #print(type(data.loc[i,'Value']))
                    #print(type(data.loc[i,'Value']) == int or type(data.loc[i,'Value']) == pd.Series)
                    
                    #print(data.loc[i,'Value'])
                    planted_values.append(data.loc[i,'Value'])
                    
                    
              
            #print(planted_values_dates)
            planted_df = pd.DataFrame(zip(planted_values_dates, planted_values), columns = ['Date','% Planted'])
            planted_df.set_index('Date', inplace = True)
            
            
            #print(planted_df.head(10))
            return planted_df
        NE_planted = getting_data('CORN', 'NE')
        
        
        NE_addresses = ['2727 W 2ND ST HASTINGS, NE 68901-4608', '4009 6TH AVE KEARNEY, NE 68845-2386', '818 FERDINAND PLZ \
        SCOTTSBLUFF, NE 69361-4401', '1202 S COTTONWOOD ST \
        NORTH PLATTE, NE 69101-6295', '120 W FAIRFIELD \
        CLAY CENTER, NE 68933-1437', '100 CENTENNIAL MALL N \
        LINCOLN, NE 68508-3803']
        

        geolocator = Nominatim(user_agent = 'young_shark_227')
        start = datetime(1979, 1,1)
        end = datetime(2022,5,18)
        
        NE_weather = {}
        
        location = geolocator.geocode(NE_addresses[4])
        
        stations = Stations()
        stations = stations.nearby(location.latitude, location.longitude)
        stations = stations.inventory('daily', datetime(1979,1,1))
        station = stations.fetch(10)
        #print(station.iloc[2])
        
        
        #print(type(NE_planted.index[0]))
        geolocator = Nominatim(user_agent = 'young_shark_227')
        
        for i in range(NE_planted.shape[0]):
            try:
                NE_planted.iloc[i, 0] = NE_planted.iloc[i,0][1] # takes away the extra stuff 
            except:
                NE_planted.iloc[i,0] = NE_planted.iloc[i,0]
        NE_planted.reset_index(inplace = True)
        NE_planted.drop_duplicates(subset = 'Date', inplace = True)
        NE_planted.set_index('Date', inplace = True)
        
        
        
        for i in range(len(NE_addresses)):
            if i in [0, 1, 4]:
                if i == 0:
                    temp = (40.9667, -98.15)
                    temp2= Point(temp[0],temp[1])
                    data = Daily(temp2, start, end)
                    NE_weather['NE'+str(i)] = data.fetch().fillna(value = 0)
                if i ==1:
                    temp = (39.55, -97.65)
                    temp2= Point(temp[0],temp[1])
                    data = Daily(temp2, start, end)
                    NE_weather['NE'+str(i)] = data.fetch().fillna(value = 0)
                if i ==4:
                    temp = (40.8333, -96.7647)
                    temp2= Point(temp[0],temp[1])
                    data = Daily(temp2, start, end)
                    NE_weather['NE'+str(i)] = data.fetch().fillna(value = 0)
                for j in list(NE_weather['NE'+str(i)].index):
                    if str(j)[0:10] not in [str(z) for z in list(NE_planted.index)]:
                        NE_weather['NE'+str(i)].drop(labels = j, axis = 0, inplace = True) # drops the weather from irrelevant dates
            else:
                location = geolocator.geocode(NE_addresses[i])
                temp = (location.latitude,location.longitude)
                
                temp2 = Point(temp[0],temp[1], )
                data = Daily(temp2, start, end)
                NE_weather['NE'+str(i)] = data.fetch().fillna(value = 0)
                for j in list(NE_weather['NE'+str(i)].index):
                    if str(j)[0:10] not in [str(z) for z in list(NE_planted.index)]:
                        NE_weather['NE'+str(i)].drop(labels = j, axis = 0, inplace = True)
            
            if i in [0, 1,4]:
                for z in range(1,6): 
                    tavg_list = []
                    prcp_list = []
                    snow_list = []
                    tsun_list = []
                    if i ==0:
                        for j in list(NE_planted.index):
                            
                            temp = (40.9667, -98.15)
                            temp2= Point(temp[0],temp[1])
                            
                            og_date = datetime.strptime(j, '%Y-%m-%d')
                            new = timedelta(days = z*7)
                            new_time = og_date - new
                            
                            data = Daily(temp2, new_time, new_time)
                            data = data.fetch().fillna(value = 0)
                            
                            tavg_list.append(data.iloc[0,0])
                            prcp_list.append(data.iloc[0,3])
                            snow_list.append(data.iloc[0,4])
                            tsun_list.append(data.iloc[0, 9])
                    
                        tavg_list.pop(359)
                        prcp_list.pop(359)
                        snow_list.pop(359)
                        tsun_list.pop(359)
                        NE_weather['NE' + str(i)]['NE'+str(i) + ' tavg ' + 't-'+ str(z)] = tavg_list
                        NE_weather['NE' + str(i)]['NE' + str(i) + ' prcp ' + 't-' + str(z)] = prcp_list
                        NE_weather['NE' + str(i)]['NE' + str(i) + ' snow ' + 't-' + str(z)] = snow_list
                        NE_weather['NE' + str(i)]['NE' + str(i) + ' tsun ' + 't-' + str(z)] = tsun_list
                    
                    
                    if i == 1:
                        for j in list(NE_planted.index):
                        
                            temp = (39.55, -97.65)
                            temp2= Point(temp[0],temp[1])
                            
                            og_date = datetime.strptime(j, '%Y-%m-%d')
                            new = timedelta(days = z*7)
                            new_time = og_date - new
                            
                            data = Daily(temp2, new_time, new_time)
                            data = data.fetch().fillna(value = 0)
                            
                            tavg_list.append(data.iloc[0,0])
                            prcp_list.append(data.iloc[0,3])
                            snow_list.append(data.iloc[0,4])
                            tsun_list.append(data.iloc[0, 9])
                            
                        tavg_list.pop(359)
                        prcp_list.pop(359)
                        snow_list.pop(359)
                        tsun_list.pop(359)
                        NE_weather['NE' + str(i)]['NE'+str(i) + ' tavg ' + 't-'+ str(z)] = tavg_list
                        NE_weather['NE' + str(i)]['NE' + str(i) + ' prcp ' + 't-' + str(z)] = prcp_list
                        NE_weather['NE' + str(i)]['NE' + str(i) + ' snow ' + 't-' + str(z)] = snow_list
                        NE_weather['NE' + str(i)]['NE' + str(i) + ' tsun ' + 't-' + str(z)] = tsun_list
                    
                    if i == 4:
                        for j in list(NE_planted.index):
                            temp = (40.8333, -96.7647)
                            temp2= Point(temp[0],temp[1])
                            og_date = datetime.strptime(j, '%Y-%m-%d')
                            new = timedelta(days = z*7)
                            new_time = og_date - new
                            
                            data = Daily(temp2, new_time, new_time)
                            data = data.fetch().fillna(value = 0)
                            
                            tavg_list.append(data.iloc[0,0])
                            prcp_list.append(data.iloc[0,3])
                            snow_list.append(data.iloc[0,4])
                            tsun_list.append(data.iloc[0, 9])
                        
                        tavg_list.pop(359)
                        prcp_list.pop(359)
                        snow_list.pop(359)
                        tsun_list.pop(359)
                        NE_weather['NE' + str(i)]['NE'+str(i) + ' tavg ' + 't-'+ str(z)] = tavg_list
                        NE_weather['NE' + str(i)]['NE' + str(i) + ' prcp ' + 't-' + str(z)] = prcp_list
                        NE_weather['NE' + str(i)]['NE' + str(i) + ' snow ' + 't-' + str(z)] = snow_list
                        NE_weather['NE' + str(i)]['NE' + str(i) + ' tsun ' + 't-' + str(z)] = tsun_list
            else:
                for z in range(1,6): 
                    tavg_list = []
                    prcp_list = []
                    snow_list = []
                    tsun_list = []
                    location = geolocator.geocode(NE_addresses[i])
                    temp = (location.latitude,location.longitude)
                    temp2 = Point(temp[0],temp[1], )
                    
                    for j in list(NE_planted.index):
                      
                        og_date = datetime.strptime(j,'%Y-%m-%d')
                        new = timedelta(days = z*7)
                        new_time = og_date - new
                        
                        data = Daily(temp2, new_time, new_time)
                        data = data.fetch().fillna(value = 0)
                        
                        tavg_list.append(data.iloc[0,0])
                        prcp_list.append(data.iloc[0,3])
                        snow_list.append(data.iloc[0,4])
                        tsun_list.append(data.iloc[0, 9])
                    
                    tavg_list.pop(359)
                    prcp_list.pop(359)
                    snow_list.pop(359)
                    tsun_list.pop(359)
                    NE_weather['NE' + str(i)]['NE'+str(i) + ' tavg ' + 't-'+ str(z)] = tavg_list
                    NE_weather['NE' + str(i)]['NE' + str(i) + ' prcp ' + 't-' + str(z)] = prcp_list
                    NE_weather['NE' + str(i)]['NE' + str(i) + ' snow ' + 't-' + str(z)] = snow_list
                    NE_weather['NE' + str(i)]['NE' + str(i) + ' tsun ' + 't-' + str(z)] = tsun_list
        
        #NE_weather_df = pd.concat([NE_weather['NE0'],NE_weather['NE1'],NE_weather['NE2'],NE_weather['NE3'],NE_weather['NE4'],NE_weather['NE5']], axis =1)
        
        for i in list(NE_weather.keys()):
            for j in list(NE_weather[i].columns):
                if 'tavg' not in str(j) and 'prcp' not in str(j) and 'snow' not in str(j) and 'tsun' not in str(j):
                    NE_weather[i].drop(labels = j, axis = 1, inplace = True) # drops irrelevant columns
            NE_weather[i].rename(columns = {'tavg': i+'tavg', 'prcp':i+'prcp','snow':i+'snow','tsun':i+'tsun'}, inplace =True) #changes column names
            
            NE_weather[i].index = NE_weather[i].index.astype('str') 
            NE_weather[i].reset_index(inplace = True)
            NE_weather[i]['time'].str[0:10]
            NE_weather[i].set_index('time', inplace = True)
            
            
        
        NE_weather_df = pd.concat([NE_weather['NE0'],NE_weather['NE1'],NE_weather['NE2'],NE_weather['NE3'],NE_weather['NE4'],NE_weather['NE5']], axis =1)
        
        NE_planted_work = NE_planted.rename(index = {'Date':'time'})
        
        df = pd.concat([NE_planted_work, NE_weather_df], axis = 1)
        
        
        df = df.iloc[:-1, :] # most recent weather data isn't in there
        
        
        df.reset_index(inplace = True)
        df.rename(columns = {'% Planted' : 'Planted', 'index': 'dt'}, inplace = True)


    else: # generic name
        
        commodity_name ='CORN' #change based on desired commodity
        state = a #for a in states loop
        year = '1900' #use GE so shouldn't matter
        
        def getting_data(commodity_name, state, year='1900'):
            
            parameters =    'source_desc=SURVEY&sector_desc=CROPS&group_desc=FIELD CROPS&commodity_desc=' + commodity_name + '&statisticcat_desc=PROGRESS&' + \
                            'agg_level_desc=STATE&state_alpha=' + state + '&year__GE=' + year + '&format=CSV'
        
            full_url = base_url_api_get + parameters
        
        
            response = requests.get(full_url)
            
            content = response.content
            data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
            
      
            relevant_columns = ['year','week_ending','state_alpha','state_name','short_desc','Value'] # for corn at least
        
            for i in list(data.columns):
                if i not in relevant_columns:
                    data.drop(i, axis =1, inplace= True)
            
            data.set_index('week_ending', inplace = True)
            data.sort_index(ascending= True, inplace = True)
            
            global planted_values_dates
            
            planted_values = []
            planted_values_dates = []
            #print(data.index)x):
                #print('PLANTED' in str(data.loc[i,'short_desc']))
            
            for i in list(data.index):
                if 'PLANTED' in str(data.loc[i, 'short_desc']):
                    planted_values_dates.append(i) #need to figure out what to do about the series
                    #if type(data.loc[i,'Value']) == pd.Series:
                     #   print(data.loc[i,'Value'][1])
                        
                    #print(type(data.loc[i,'Value']))
                    #print(type(data.loc[i,'Value']) == int or type(data.loc[i,'Value']) == pd.Series)
                    
                    #print(data.loc[i,'Value'])
                    planted_values.append(data.loc[i,'Value'])
                    
                    
              
            #print(planted_values_dates)
            planted_df = pd.DataFrame(zip(planted_values_dates, planted_values), columns = ['Date','Planted'])
            planted_df.set_index('Date', inplace = True)
            
            
            #print(planted_df.head(10))
            return planted_df
        planted_df = getting_data('CORN', a) #i represents the state
        #print(planted_df)
        
        NE_addresses = ['2727 W 2ND ST HASTINGS, NE 68901-4608', '4009 6TH AVE KEARNEY, NE 68845-2386', '818 FERDINAND PLZ \
        SCOTTSBLUFF, NE 69361-4401', '1202 S COTTONWOOD ST \
        NORTH PLATTE, NE 69101-6295', '120 W FAIRFIELD \
        CLAY CENTER, NE 68933-1437', '100 CENTENNIAL MALL N \
        LINCOLN, NE 68508-3803']
        
        if a =='IA':
            addresses = ['1621 LAKE AVE STORM LAKE, IA 50588-1913', '1301 6TH AVE N HUMBOLDT, IA 50548-1150',
                         '2296 OIL WELL RD DECORAH, IA 52101-7369','1100 12TH ST SW LE MARS, IA 51031-3034',
                         '840 BROOKS RD IOWA FALLS, IA 50126-8008',
                         '2505 N BROADWAY ST RED OAK, IA 51566-1078', '1701 S B ST ALBIA, IA 52531-2685', 
                         '605 S 23RD ST FAIRFIELD, IA 52556-4212']

        if a == 'IL':
            addresses = ['225 N MAIN ST ELIZABETH, IL 61028-8802', '1350 W PRAIRIE DR SYCAMORE, IL 60178-3166',
                         '1201 GOUGAR RD NEW LENOX, IL 60451-9748','338 S 36TH ST QUINCY, IL 62301-5807',
                         '2031 MASCOUTAH AVE BELLEVILLE, IL 62220-3418','221 WITHERS DR MT VERNON, IL 62864-8175', 
                         '1105 W MAIN ST CARMI, IL 62821-1380']
            
        if a == 'MN':
            addresses = ['105 S DIVISION ST WARREN, MN 56762-1406','4850 MILLER TRUNK HWY DULUTH, MN 55811-1506','1615 30TH AVE S MOORHEAD, MN 56560',
                         '110 2ND ST S WAITE PARK, MN 56387-1665', '1424 E COLLEGE DR MARSHALL, MN 56258-2090',
                        '1160 VICTORY DR MANKATO, MN 56001-5307','1810 30TH ST NW FARIBAULT, MN 55021-1843']
        
        if a == 'IN':
            addresses = ['211 E DREXEL PKWY RENSSELAER, IN 47978-7294','17746 COUNTY ROAD 34 GOSHEN, IN 46528-9261',
                         '036 LEBANON RD CRAWFORDSVILLE, IN 47933-2143','195 MEADOW DR DANVILLE, IN 46122-1413',
                        '975 E WASHINGTON ST WINCHESTER, IN 47394-9221']
        
        if a == 'KS':
            addresses = ['210 W 10TH ST GOODLAND, KS 67735-2836', '105 W SOUTH ST MANKATO, KS 66956-2236', 
                         '1310 OREGON ST HIAWATHA, KS 66434-2203', '2106 E SPRUCE ST GARDEN CITY, KS 67846-6362',
                         '313 CROSS ST BURLINGTON, KS 66839-1190',
                         '300 E COUNTRY RD COLUMBUS, KS 66725-1809']
            
        if a == 'SD': #SD LOCATIONS ARE BAD
            addresses = ['414 E STUMER RD RAPID CITY, SD 57701-6414','205 6TH ST BROOKINGS, SD 57006-1406', 
                         '1717 N LINCOLN AVE PIERRE, SD 57501-2398','2408 E BENSON RD SIOUX FALLS, SD 57104-7018',
                         '1820 N KIMBALL ST MITCHELL, SD 57301-1114']
            
        if a == 'OH':
            addresses = ['1800 N PERRY ST OTTAWA, OH 45875-1199','1800 N PERRY ST OTTAWA, OH 45875-1199', '1800 N PERRY ST OTTAWA, OH 45875-1199']#'7868 COUNTY ROAD 140 FINDLAY, OH 45840-1898']#
            #            '111 JACKSON PIKE ST 1569 GALLIPOLIS, OH 45631-1568', '10025 AMITY RD BROOKVILLE, OH 45309-9399']
            #'1834 S LINCOLN AVE SALEM, OH 44460-4393' - erors ;;; '777 COLUMBUS AVE STE 3A LEBANON, OH 45036-1682';;'11752 STATE ROUTE 104 WAVERLY, OH 45690-9660'
        if a == 'WI':
            addresses = ['800 N FRONT ST SPOONER, WI 54801-1350', 
                        '603A LAKELAND RD SHAWANO, WI 54166-3843','390 RED CEDAR ST MENOMONIE, WI 54751-2265', 
                        'W6529 FOREST AVE FOND DU LAC, WI 54937-9489', '1926 EASTERN AVE PLYMOUTH, WI 53073-4263',
                        '150 W ALONA LN LANCASTER, WI 53813-2182', '5201 FEN OAK DRIVE MADISON, WI 53718'
                        'W6529 FOREST AVE FOND DU LAC, WI 54937-9489']
        #doesn't work: '925 DONALD ST RM 101 MEDFORD, WI 54451-2099',
        if a == 'MO':
            addresses = ['1101 S POLK ST MAYSVILLE, MO 64469-4042', '23487 ECLIPSE DR MILAN, MO 63556-2877', 
                         '502 S WASHINGTON ST MONTICELLO, MO 63457-9715','625 W NURSERY ST BUTLER, MO 64730-0192', 
                         '1242 DEADRA DR LEBANON, MO 65536-4669','812 PROGRESS DR FARMINGTON, MO 63640-9157']
        
        geolocator = Nominatim(user_agent = 'young_shark_227')

        weather = {}
        start = datetime(1986,5,4) 
        end = datetime(2022,6,1)
            
#CODE TO GET STATIONS    
#        if a == 'IA':
#            location = geolocator.geocode(address)
#        
#       location = geolocator.geocode(address)
#temp = (location.latitude,location.longitude)
#
#stations = Stations()
#stations= stations.nearby(temp[0],temp[1])
#
#locations= stations.fetch(5)
#
#lat_list = []
#long_list= []
#
#for z in range(5):
#    lat_list.append(locations.iloc[z, 5])
#    long_list.append(locations.iloc[z,6])

        for i in range(planted_df.shape[0]):
            try:
                planted_df.iloc[i, 0] = planted_df.iloc[i,0][1] # takes away the extra stuff 
            except:
                planted_df.iloc[i,0] = planted_df.iloc[i,0]
        planted_df.reset_index(inplace = True)
        planted_df.drop_duplicates(subset = 'Date', inplace = True)
        planted_df.set_index('Date', inplace = True)
        

        #SUCESSFULLY GIVES US THE PLANTED VALUES^^
        
        long_list = []
        lat_list = []

        start = datetime(1986,5,4) 
        end = datetime(2022,6,1)
       # test_lst = []
        for i in range(len(addresses)):
            
            
            
            if a == 'IA':
                
                
                
                start = datetime(1986,5,4)
                
                if i == 6:
                    location = geolocator.geocode(addresses[i])
                    long_list.append(location.longitude)
                    lat_list.append(location.latitude)
                    temp2 = Point(lat_list[i], long_list[i], )
                    data = Daily(temp2, start, end)
                    
                    weather[a + str(i)] = data.fetch().fillna(value = 0)
                    for j in list(weather[a+str(i)].index):
                        if str(j)[0:10] not in [str(z) for z in list(planted_df.index)]:
                            weather[a+str(i)].drop(labels = j, axis = 0, inplace = True)
                    continue
                else:
                    k = 0
                    location = geolocator.geocode(addresses[4])
                    stations = Stations()
                    locations = stations.fetch(8)
                    lat_list.append(locations.iloc[k,5])
                    long_list.append(locations.iloc[k,6])
                    k+= 1
                    
                    temp2 = Point(lat_list[i], long_list[i])
                    data = Daily(temp2, start, end)
                    
                    weather[a + str(i)] = data.fetch().fillna(value = 0)
                    for j in list(weather[a+str(i)].index):
                        if str(j)[0:10] not in [str(z) for z in list(planted_df.index)]:
                            weather[a+str(i)].drop(labels = j, axis = 0, inplace = True)
                    
                    continue
            
            
            if a == 'IL':
                
                
                if i in [0,3,4]:
                    location = geolocator.geocode(addresses[i])
                    long_list.append(location.longitude)
                    lat_list.append(location.latitude)
                    temp2 = Point(lat_list[i], long_list[i], )
                    data = Daily(temp2, start, end)
                    location = geolocator.geocode(addresses[i])
                    temp = (location.latitude,location.longitude)
                    weather[a + str(i)] = data.fetch().fillna(value = 0)
                    for j in list(weather[a+str(i)].index):
                        if str(j)[0:10] not in [str(z) for z in list(planted_df.index)]:
                            weather[a+str(i)].drop(labels = j, axis = 0, inplace = True)
                    continue
                else:
                    k = 0
                    location = geolocator.geocode(addresses[5])
                    stations = Stations()
                    locations = stations.fetch(4)
                    lat_list.append(locations.iloc[k,5])
                    long_list.append(locations.iloc[k,6])
                    k+= 1
                    
                    temp2 = Point(lat_list[i], long_list[i])
                    data = Daily(temp2, start, end)
                    
                    weather[a + str(i)] = data.fetch().fillna(value = 0)
                    for j in list(weather[a+str(i)].index):
                        if str(j)[0:10] not in [str(z) for z in list(planted_df.index)]:
                            weather[a+str(i)].drop(labels = j, axis = 0, inplace = True)
                    
                    
                    for z in range(1,6):
                        weather[a+str(i)+' t- '+str(z)] = weather[a+str(i)].shift(-z).fillna(value = 0)
                        for j in list(weather[a+str(i)].index):
                            if str(j)[0:10] not in [str(z) for z in list(planted_df.index)]:
                                weather[a+str(i) + 't- '+str(z)].drop(labels = j, axis = 0, inplace = True)
                    
                    for j in list(weather[a+str(i)].index):
                        if str(j)[0:10] not in [str(z) for z in list(planted_df.index)]:
                            weather[a+str(i)].drop(labels = j, axis = 0, inplace = True)
                        
       

            else:
                
                location = geolocator.geocode(addresses[2])
                stations = Stations()
                locations = stations.fetch(8)
                
                lat_list.append(locations.iloc[i,5])
                long_list.append(locations.iloc[i,6])
                
                temp2 = Point(lat_list[i], long_list[i])
                data = Daily(temp2, start, end)
                
                weather[a + str(i)] = data.fetch().fillna(value = 0)
                
                for z in range(1,6):
                    
                    weather[a+str(i)+' t- '+str(z)] = weather[a+str(i)].shift(-z).fillna(value = 0)
                    for j in list(weather[a+str(i)].index):
                        if str(j)[0:10] not in [str(z) for z in list(planted_df.index)]:
                            weather[a+str(i) + ' t- '+str(z)].drop(labels = j, axis = 0, inplace = True)
                
                for j in list(weather[a+str(i)].index):
                    if str(j)[0:10] not in [str(z) for z in list(planted_df.index)]:
                        weather[a+str(i)].drop(labels = j, axis = 0, inplace = True)
                        

            
            
        df = pd.concat(weather.values(), axis =1)
        
        @print(df)
        
        df.to_pickle("//tedfil01/DataDropDEV/PythonPOC/CQG/usdadata.pkl")
        df = pd.read_pickle("//tedfil01/DataDropDEV/PythonPOC/CQG/usdadata.pkl")
        output.close()
        
        db.dataFrameToSQL('RiskPOC', df, 'USDA_ML_Data', "//tedfil01/DataDropDEV/PythonPOC/Upload_CSVs/usdamldata.csv", True)

                
                
                #continue
            #test_lst.append(location)
            
#            temp2 = Point(temp[0],temp[1], )
#            data = Daily(temp2, start, end)
#            weather[a+str(i)] = data.fetch().fillna(value = 0)
#            for j in list(weather[a+str(i)].index):
#                if str(j)[0:10] not in [str(z) for z in list(planted_df.index)]:
#                    weather[a+str(i)].drop(labels = j, axis = 0, inplace = True)

#           
#        for i in list(weather.keys()):
#            for j in list(weather[i].columns):
#                if 'tavg' not in str(j) and 'prcp' not in str(j) and 'snow' not in str(j):
#                    weather[i].drop(labels = j, axis = 1, inplace = True) # drops irrelevant columns
#        
        
#        for i in #print(weather)
#            #print(long_list)
#            for z in range(1,6): 
#                tavg_list = []
#                prcp_list = []
#                snow_list = []
#                tsun_list = []
#        
#                for k in range(len(long_list)): ##returning 0 i guess
#                    
#                    for j in list(planted_df.index):
#                        
#                        temp = (lat_list[k], long_list[k])
#                        temp2= Point(temp[0],temp[1])
#                        
#                        og_date = datetime.strptime(j, '%Y-%m-%d')
#                        new = timedelta(days = z*7)
#                        new_time = og_date - new
#                        
#                        
#                        
#                        
#                        data = Daily(temp2, new_time, new_time)
#                        data = data.fetch().fillna(value = 0)
#                        
#                        tavg_list.append(data.iloc[0,0])
#                        prcp_list.append(data.iloc[0,3])
#                        snow_list.append(data.iloc[0,4])
#                        tsun_list.append(data.iloc[0, 9])
#                
##don't think we need this                    tavg_list.pop(359)
##                    prcp_list.pop(359)
##                    snow_list.pop(359)
##                    tsun_list.pop(359)
#                    weather[a + str(i)][a+str(i) + ' tavg ' + 't-'+ str(z)] = tavg_list
#                    weather[a + str(i)][a + str(i) + ' prcp ' + 't-' + str(z)] = prcp_list
#                    weather[a + str(i)][a + str(i) + ' snow ' + 't-' + str(z)] = snow_list
#                    weather[a + str(i)][a + str(i) + ' tsun ' + 't-' + str(z)] = tsun_list


        #print(weather)

#        tavg_list = []
#        prcp_list = []
#        snow_list = []
#        tsun_list = []
#        
#        for i in list(planted_df.index):
#            tavg_list.append(data.iloc[0,0])
#            prcp_list.append(data.iloc[0,3])
#            snow_list.append(data.iloc[0,4])
#
#            weather[a + str(i)][a+str(i) + ' tavg ' + 't-'+ str(z)] = tavg_list
#            weather[a + str(i)][a + str(i) + ' prcp ' + 't-' + str(z)] = prcp_list
#            weather[a + str(i)][a + str(i) + ' snow ' + 't-' + str(z)] = snow_list
#
#        
        
        

#        for i in range(len(NE_addresses)):
#            if i in [0, 1, 4]:
#                if i == 0:
#                    temp = (40.9667, -98.15)
#                    temp2= Point(temp[0],temp[1])
#                    data = Daily(temp2, start, end)
#                    NE_weather['NE'+str(i)] = data.fetch().fillna(value = 0)
#                if i ==1:
#                    temp = (39.55, -97.65)
#                    temp2= Point(temp[0],temp[1])
#                    data = Daily(temp2, start, end)
#                    NE_weather['NE'+str(i)] = data.fetch().fillna(value = 0)
#                if i ==4:
#                    temp = (40.8333, -96.7647)
#                    temp2= Point(temp[0],temp[1])
#                    data = Daily(temp2, start, end)
#                    NE_weather['NE'+str(i)] = data.fetch().fillna(value = 0)
#                for j in list(NE_weather['NE'+str(i)].index):
#                    if str(j)[0:10] not in [str(z) for z in list(NE_planted.index)]:
#                        NE_weather['NE'+str(i)].drop(labels = j, axis = 0, inplace = True) # drops the weather from irrelevant dates
#            else:
#                location = geolocator.geocode(NE_addresses[i])
#                temp = (location.latitude,location.longitude)
#                
#                temp2 = Point(temp[0],temp[1], )
#                data = Daily(temp2, start, end)
#                NE_weather['NE'+str(i)] = data.fetch().fillna(value = 0)
#                for j in list(NE_weather['NE'+str(i)].index):
#                    if str(j)[0:10] not in [str(z) for z in list(NE_planted.index)]:
#                        NE_weather['NE'+str(i)].drop(labels = j, axis = 0, inplace = True)
#            
#            if i in [0, 1,4]:
#                for z in range(1,6): 
#                    tavg_list = []
#                    prcp_list = []
#                    snow_list = []
#                    tsun_list = []
#                    if i ==0:
#                        for j in list(NE_planted.index):
#                            
#                            temp = (40.9667, -98.15)
#                            temp2= Point(temp[0],temp[1])
#                            
#                            og_date = datetime.strptime(j, '%Y-%m-%d')
#                            new = timedelta(days = z*7)
#                            new_time = og_date - new
#                            
#                            data = Daily(temp2, new_time, new_time)
#                            data = data.fetch().fillna(value = 0)
#                            
#                            tavg_list.append(data.iloc[0,0])
#                            prcp_list.append(data.iloc[0,3])
#                            snow_list.append(data.iloc[0,4])
#                            tsun_list.append(data.iloc[0, 9])
#                    
#                        tavg_list.pop(359)
#                        prcp_list.pop(359)
#                        snow_list.pop(359)
#                        tsun_list.pop(359)
#                        NE_weather['NE' + str(i)]['NE'+str(i) + ' tavg ' + 't-'+ str(z)] = tavg_list
#                        NE_weather['NE' + str(i)]['NE' + str(i) + ' prcp ' + 't-' + str(z)] = prcp_list
#                        NE_weather['NE' + str(i)]['NE' + str(i) + ' snow ' + 't-' + str(z)] = snow_list
#                        NE_weather['NE' + str(i)]['NE' + str(i) + ' tsun ' + 't-' + str(z)] = tsun_list
#                    
#                    
#                    if i == 1:
#                        for j in list(NE_planted.index):
#                        
#                            temp = (39.55, -97.65)
#                            temp2= Point(temp[0],temp[1])
#                            
#                            og_date = datetime.strptime(j, '%Y-%m-%d')
#                            new = timedelta(days = z*7)
#                            new_time = og_date - new
#                            
#                            data = Daily(temp2, new_time, new_time)
#                            data = data.fetch().fillna(value = 0)
#                            
#                            tavg_list.append(data.iloc[0,0])
#                            prcp_list.append(data.iloc[0,3])
#                            snow_list.append(data.iloc[0,4])
#                            tsun_list.append(data.iloc[0, 9])
#                            
#                        tavg_list.pop(359)
#                        prcp_list.pop(359)
#                        snow_list.pop(359)
#                        tsun_list.pop(359)
#                        NE_weather['NE' + str(i)]['NE'+str(i) + ' tavg ' + 't-'+ str(z)] = tavg_list
#                        NE_weather['NE' + str(i)]['NE' + str(i) + ' prcp ' + 't-' + str(z)] = prcp_list
#                        NE_weather['NE' + str(i)]['NE' + str(i) + ' snow ' + 't-' + str(z)] = snow_list
#                        NE_weather['NE' + str(i)]['NE' + str(i) + ' tsun ' + 't-' + str(z)] = tsun_list
#                    
#                    if i == 4:
#                        for j in list(NE_planted.index):
#                            temp = (40.8333, -96.7647)
#                            temp2= Point(temp[0],temp[1])
#                            og_date = datetime.strptime(j, '%Y-%m-%d')
#                            new = timedelta(days = z*7)
#                            new_time = og_date - new
#                            
#                            data = Daily(temp2, new_time, new_time)
#                            data = data.fetch().fillna(value = 0)
#                            
#                            tavg_list.append(data.iloc[0,0])
#                            prcp_list.append(data.iloc[0,3])
#                            snow_list.append(data.iloc[0,4])
#                            tsun_list.append(data.iloc[0, 9])
#                        
#                        tavg_list.pop(359)
#                        prcp_list.pop(359)
#                        snow_list.pop(359)
#                        tsun_list.pop(359)
#                        NE_weather['NE' + str(i)]['NE'+str(i) + ' tavg ' + 't-'+ str(z)] = tavg_list
#                        NE_weather['NE' + str(i)]['NE' + str(i) + ' prcp ' + 't-' + str(z)] = prcp_list
#                        NE_weather['NE' + str(i)]['NE' + str(i) + ' snow ' + 't-' + str(z)] = snow_list
#                        NE_weather['NE' + str(i)]['NE' + str(i) + ' tsun ' + 't-' + str(z)] = tsun_list
#            else:
#                for z in range(1,6): 
#                    tavg_list = []
#                    prcp_list = []
#                    snow_list = []
#                    tsun_list = []
#                    location = geolocator.geocode(NE_addresses[i])
#                    temp = (location.latitude,location.longitude)
#                    temp2 = Point(temp[0],temp[1], )
#                    
#                    for j in list(NE_planted.index):
#                      
#                        og_date = datetime.strptime(j,'%Y-%m-%d')
#                        new = timedelta(days = z*7)
#                        new_time = og_date - new
#                        
#                        data = Daily(temp2, new_time, new_time)
#                        data = data.fetch().fillna(value = 0)
#                        
#                        tavg_list.append(data.iloc[0,0])
#                        prcp_list.append(data.iloc[0,3])
#                        snow_list.append(data.iloc[0,4])
#                        tsun_list.append(data.iloc[0, 9])
#                    
#                    tavg_list.pop(359)
#                    prcp_list.pop(359)
#                    snow_list.pop(359)
#                    tsun_list.pop(359)
#                    NE_weather['NE' + str(i)]['NE'+str(i) + ' tavg ' + 't-'+ str(z)] = tavg_list
#                    NE_weather['NE' + str(i)]['NE' + str(i) + ' prcp ' + 't-' + str(z)] = prcp_list
#                    NE_weather['NE' + str(i)]['NE' + str(i) + ' snow ' + 't-' + str(z)] = snow_list
#                    NE_weather['NE' + str(i)]['NE' + str(i) + ' tsun ' + 't-' + str(z)] = tsun_list
#        
#        #NE_weather_df = pd.concat([NE_weather['NE0'],NE_weather['NE1'],NE_weather['NE2'],NE_weather['NE3'],NE_weather['NE4'],NE_weather['NE5']], axis =1)
#        
#        for i in list(NE_weather.keys()):
#            for j in list(NE_weather[i].columns):
#                if 'tavg' not in str(j) and 'prcp' not in str(j) and 'snow' not in str(j) and 'tsun' not in str(j):
#                    NE_weather[i].drop(labels = j, axis = 1, inplace = True) # drops irrelevant columns
#            NE_weather[i].rename(columns = {'tavg': i+'tavg', 'prcp':i+'prcp','snow':i+'snow','tsun':i+'tsun'}, inplace =True) #changes column names
#            
#            NE_weather[i].index = NE_weather[i].index.astype('str') 
#            NE_weather[i].reset_index(inplace = True)
#            NE_weather[i]['time'].str[0:10]
#            NE_weather[i].set_index('time', inplace = True)
#            
#            
#        
#        NE_weather_df = pd.concat([NE_weather['NE0'],NE_weather['NE1'],NE_weather['NE2'],NE_weather['NE3'],NE_weather['NE4'],NE_weather['NE5']], axis =1)
#        
#        NE_planted_work = NE_planted.rename(index = {'Date':'time'})
#        
#        df = pd.concat([NE_planted_work, NE_weather_df], axis = 1)
#        
#        
#        df = df.iloc[:-1, :] # most recent weather data isn't in there
#        
#        
#        df.reset_index(inplace = True)
#        df.rename(columns = {'% Planted' : 'Planted', 'index': 'dt'}, inplace = True)

"""
#PICKLE STUFF

df.to_pickle("//tedfil01/DataDropDEV/PythonPOC/CQG/usdadata.pkl")
df = pd. read_pickle("//tedfil01/DataDropDEV/PythonPOC/CQG/usdadata.pkl")
#output.close()

db.dataFrameToSQL('RiskPOC', df, 'USDA_ML_Data', "//tedfil01/DataDropDEV/PythonPOC/Upload_CSVs/usdamldata.csv", True)

"""
"""


#MACHINE LEARNING PART - predicting the % planted of the next week given the past conditions
#data pre-processing - need to convert time series into supervised learning

#n_in - how many lag observations you have; should do some sort of feature selection on this where have n_out = 4, 3, 2,1,etc.
def conv_to_sup_learn(data,n_in = 1, n_out =1, dropnan = True): #n_out - how many t steps into the future you want to forecast (n+1, etc.)
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    shifted_feats, col_names = list(), list()
    
    #input sequence
    
    for i in range(n_in, 0, -1):
        shifted_feats.append(df.shift(i)) # shifts so it is n-5, n-4,...n features 
        col_names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)] #sets col names -- var1(t-5), var2(t_5), ..., 
    
    #forecasting sequence (t, t+1, t+n...)
    
    for i in range(0, n_out):
        shifted_feats.append(df.shift(-i))
        if i ==0:
            col_names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            col_names += [('var%d(t+%d)' %(j+1, i)) for j in range(n_vars)]
            
    total = pd.concat(shifted_feats, axis = 1)
    
    total.columns = col_names
    if dropnan:
        total.dropna(inplace = True)
    
    return total
        
values = df.values
encod = LabelEncoder()
values[:, 4] = encod.fit_transform(values[:,4])



values = values.astype('float32')
#normalizes data
scaler = MinMaxScaler(feature_range = (0,1))
values = scaler.fit_transform(values)

#supervised learning
reframed = conv_to_sup_learn(values, 1,1)



for i in list(reframed.columns):
    if '(t-1)' not in str(i):
        if '1(t)' not in str(i):
            reframed.drop(i, inplace = True, axis = 1)
        else:
            if '11(t)' in str(i) or '21(t)' in str(i):
                reframed.drop(i, inplace = True, axis =1)
            

#train/test sets
train_size = int(len(values)* 0.9)
test_size = len(values) - train_size
train, test = values[0:train_size, :], values[train_size:len(values), :]

train_X, train_Y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:,-1]

#reshaping into [samples, timesteps, faetures]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))

#LSTM model
model = Sequential()


model.add(LSTM(80, activation='relu', return_sequences = True, input_shape = (train_X.shape[1], train_X.shape[2])))
model.add(LSTM(50, activation='relu'))
model.add(Dense(20, activation='linear'))
opt = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer=opt, loss='mse')




#fitting
history = model.fit(train_X, train_Y, epochs = 100, batch_size = 72, validation_data = (test_X, test_y), verbose = 2, shuffle = False)


#plotting
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'test')

plt.legend()
plt.ylim([0,1])

plt.show()

"""

"""
#prediction
y_predict = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

#inverse scale
inv_ypredict = np.concatenate((y_predict, test_X[:, 1:]), axis =1)
un_scaled = scaler.inverse_transform(inv_ypredict)

inverse_y_predict = un_scaled[:,0]

#invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis =1)
inv_y = scaler.inverse_transform(inv_y)

inv_y = inv_y[:,0]

rmse = sqrt(mean_squared_error(inv_y, inverse_y_predict))

print('Test RMSE: %.3f' % rmse)
"""




"""
#predicition
y_pred = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_features))

inv_ypred = pd.concat((y_pred, test_X[:, -24:]), axis =1)
"""
"""

#gets train size to be first 80% then values to be the remaining 80%
train_size = int(len(values)* 0.8)
test_size = len(values) - train_size
train, test = values[0:train_size, :], values[train_size:len(values), :]


def split_seq(data, look_back = 1): #adjust look backperiod with cross-validation -- switch this up to get the data done correctly
    X_data, Y_data = list(), list()
    
    for i in range(len(values)-look_back - 1):
        a = values[i:(i+look_back), 0]
        X_data.append(a)
        Y_data.append(values[i+look_back, 0])
    
    return np.array(X_data), np.array(Y_data)



look_back = 1
trainX, trainY = split_seq(train, look_back)
testX, testY = split_seq(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0],1, trainX.shape[0]))
testX = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))


#creating/fitting the LSTM network
model = Sequential()
model.add(LSTM(80, activation='relu', return_sequences=True, input_shape=(1, look_back)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(20, activation='linear'))
opt = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer=opt, loss='mse')




model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimzer = 'adam')
model.fit(trainX, trainY, epochs = 100, batch_size =1, verbose = None)


#Model Results
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#re-vert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

#RMSE calculations
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %2f RMSE' %(trainScore))
testScore = math.sqrt(mean_squared_error(testY[0],testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))



#maybe convert NE_weather to NumPy arrays: data = np.array(df).reshape(-1,1) - will help it run faster
scale = MinMaxScaler(feature_range=(0,1))
scaledData = scale.fit_transform(data)

rmse = [3,0]
i = 2 # number of weeks back


while abs(rmse[-1] - rmse[-2]) >1:
    dataTable = np.zeros((len(data) - i, i))
    predicted = scaledData[i:, :]
    
    for i in range
 


"""

