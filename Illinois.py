# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:38:47 2022

@author: ZAK0131
"""
import requests
import io
import json
import csv
import urllib
##TEST open weather API


#Scrape weather underground:
from bs4 import BeautifulSoup as BS
from selenium import webdriver
from functools import reduce
import pandas as pd
import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from pyvirtualdisplay import Display


def render_page(url):
    #display = Display(visible=0, size=(800, 600))
    #display.start()
    option = webdriver.ChromeOptions()
    option.add_argument('headless')
    driver = webdriver.Chrome(r'//tedfil01/DataDropDEV/PythonPOC/Amrith/chromedriver.exe', options = option)
    driver.get(url)
    time.sleep(3)
    r = driver.page_source
    driver.quit()
    #display.stop()
    return r

def scraper(page, dates):
    output = pd.DataFrame()
    
    
    for d in dates:
        try:
            url = str(str(page) + str(d))
        
            r = render_page(url)
        
            soup = BS(r, "html.parser")
            container = soup.find('lib-city-history-observation')
            check = container.find('tbody')
        
            data = []
        
            for c in check.find_all('tr', class_='ng-star-inserted'):
                for i in c.find_all('td', class_='ng-star-inserted'):
                    trial = i.text
                    trial = trial.strip('  ')
                    data.append(trial)
        
            if round(len(data) / 17 - 1) == 31:
                Temperature = pd.DataFrame([data[32:128][x:x + 3] for x in range(0, len(data[32:128]), 3)][1:],
                                           columns=['Temp_max', 'Temp_avg', 'Temp_min'])
                Dew_Point = pd.DataFrame([data[128:224][x:x + 3] for x in range(0, len(data[128:224]), 3)][1:],
                                         columns=['Dew_max', 'Dew_avg', 'Dew_min'])
                Humidity = pd.DataFrame([data[224:320][x:x + 3] for x in range(0, len(data[224:320]), 3)][1:],
                                        columns=['Hum_max', 'Hum_avg', 'Hum_min'])
                Wind = pd.DataFrame([data[320:416][x:x + 3] for x in range(0, len(data[320:416]), 3)][1:],
                                    columns=['Wind_max', 'Wind_avg', 'Wind_min'])
                Pressure = pd.DataFrame([data[416:512][x:x + 3] for x in range(0, len(data[416:512]), 3)][1:],
                                        columns=['Pres_max', 'Pres_avg', 'Pres_min'])
                Date = pd.DataFrame(data[:32][1:], columns=data[:1])
                Precipitation = pd.DataFrame(data[512:][1:], columns=['Precipitation'])
                print(str(str(d) + ' finished!'))
            elif round(len(data) / 17 - 1) == 28:
                Temperature = pd.DataFrame([data[29:116][x:x + 3] for x in range(0, len(data[29:116]), 3)][1:],
                                           columns=['Temp_max', 'Temp_avg', 'Temp_min'])
                Dew_Point = pd.DataFrame([data[116:203][x:x + 3] for x in range(0, len(data[116:203]), 3)][1:],
                                         columns=['Dew_max', 'Dew_avg', 'Dew_min'])
                Humidity = pd.DataFrame([data[203:290][x:x + 3] for x in range(0, len(data[203:290]), 3)][1:],
                                        columns=['Hum_max', 'Hum_avg', 'Hum_min'])
                Wind = pd.DataFrame([data[290:377][x:x + 3] for x in range(0, len(data[290:377]), 3)][1:],
                                    columns=['Wind_max', 'Wind_avg', 'Wind_min'])
                Pressure = pd.DataFrame([data[377:464][x:x + 3] for x in range(0, len(data[377:463]), 3)][1:],
                                        columns=['Pres_max', 'Pres_avg', 'Pres_min'])
                Date = pd.DataFrame(data[:29][1:], columns=data[:1])
                Precipitation = pd.DataFrame(data[464:][1:], columns=['Precipitation'])
                print(str(str(d) + ' finished!'))
            elif round(len(data) / 17 - 1) == 29:
                Temperature = pd.DataFrame([data[30:120][x:x + 3] for x in range(0, len(data[30:120]), 3)][1:],
                                           columns=['Temp_max', 'Temp_avg', 'Temp_min'])
                Dew_Point = pd.DataFrame([data[120:210][x:x + 3] for x in range(0, len(data[120:210]), 3)][1:],
                                         columns=['Dew_max', 'Dew_avg', 'Dew_min'])
                Humidity = pd.DataFrame([data[210:300][x:x + 3] for x in range(0, len(data[210:300]), 3)][1:],
                                        columns=['Hum_max', 'Hum_avg', 'Hum_min'])
                Wind = pd.DataFrame([data[300:390][x:x + 3] for x in range(0, len(data[300:390]), 3)][1:],
                                    columns=['Wind_max', 'Wind_avg', 'Wind_min'])
                Pressure = pd.DataFrame([data[390:480][x:x + 3] for x in range(0, len(data[390:480]), 3)][1:],
                                        columns=['Pres_max', 'Pres_avg', 'Pres_min'])
                Date = pd.DataFrame(data[:30][1:], columns=data[:1])
                Precipitation = pd.DataFrame(data[480:][1:], columns=['Precipitation'])
                print(str(str(d) + ' finished!'))
            elif round(len(data) / 17 - 1) == 30:
                Temperature = pd.DataFrame([data[31:124][x:x + 3] for x in range(0, len(data[31:124]), 3)][1:],
                                           columns=['Temp_max', 'Temp_avg', 'Temp_min'])
                Dew_Point = pd.DataFrame([data[124:217][x:x + 3] for x in range(0, len(data[124:217]), 3)][1:],
                                         columns=['Dew_max', 'Dew_avg', 'Dew_min'])
                Humidity = pd.DataFrame([data[217:310][x:x + 3] for x in range(0, len(data[217:310]), 3)][1:],
                                        columns=['Hum_max', 'Hum_avg', 'Hum_min'])
                Wind = pd.DataFrame([data[310:403][x:x + 3] for x in range(0, len(data[310:403]), 3)][1:],
                                    columns=['Wind_max', 'Wind_avg', 'Wind_min'])
                Pressure = pd.DataFrame([data[403:496][x:x + 3] for x in range(0, len(data[403:496]), 3)][1:],
                                        columns=['Pres_max', 'Pres_avg', 'Pres_min'])
                Date = pd.DataFrame(data[:31][1:], columns=data[:1])
                Precipitation = pd.DataFrame(data[496:][1:], columns=['Precipitation'])
                print(str(str(d) + ' finished!'))
            else:
                print('Data not in normal length')
        
            dfs = [Date, Temperature, Dew_Point, Humidity, Wind, Pressure, Precipitation]
        
            df_final = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)
        
            df_final['Date'] = str(d) + "-" + df_final.iloc[:, :1].astype(str)
        
            output = output.append(df_final)
        except:
            print("date not working")
            bad_dates.append(d)
    print('Scraper done!')
    
    output = output[['Temp_avg', 'Temp_min', 'Dew_max', 'Dew_avg', 'Dew_min', 'Hum_max',
                     'Hum_avg', 'Hum_min', 'Wind_max', 'Wind_avg', 'Wind_min', 'Pres_max',
                     'Pres_avg', 'Pres_min', 'Precipitation', 'Date']]
    
    return output

total_data = pd.DataFrame()



bad_dates = []
url_list = ['https://www.wunderground.com/history/monthly/us/il/pleasant-plains/KSPI/date/', 
            'https://www.wunderground.com/history/monthly/us/ia/dubuque/KDBQ/date/']

next_urls = ['https://www.wunderground.com/history/monthly/us/il/marion/KMWA/date/',
             'https://www.wunderground.com/history/monthly/us/mo/st.-louis/KSTL/date/',
             'https://www.wunderground.com/history/monthly/us/il/peoria/KPIA/date/',
             'https://www.wunderground.com/history/monthly/us/il/tampico/KMLI/date/']

next_urls1 = ['https://www.wunderground.com/history/monthly/us/il/champaign/KCMI/date/',
              'https://www.wunderground.com/history/monthly/us/il/rockford/KRFD/date/',
              'https://www.wunderground.com/history/monthly/us/il/lincoln/KAAA/date/',
              'https://www.wunderground.com/history/monthly/us/il/godfrey/KSTL/date/']

for i in next_urls1:
    print(i)
    dates = dates_list
    page = i
    df_output = scraper(page,dates)
    df_output['station'] = i
    total_data = total_data.append(df_output)



#total_data.to_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/IL_WeatherU_scrapped.csv')


from datetime import datetime, date, timedelta

end = datetime(2022, 6, 1)
start = datetime(1979, 12, 1)
datelist = pd.date_range(start, end, freq = "M")
datelist = pd.DataFrame({'date':datelist})

datelist = datelist[(datelist['date'].dt.month == 3) | (datelist['date'].dt.month == 4) | (datelist['date'].dt.month == 5) | (datelist['date'].dt.month == 6)]


datelist.date.strftime('%Y-%m')

dates_list = []
for i, j in datelist.iterrows():
    print(j)
    j = j.iat[0].strftime('%Y-%m')
    dates_list.append(j)





total_data = total_data.reset_index()

total_data = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/IL_WeatherU_scrapped.csv')

total_data = total_data[total_data.Date != '1984-04-31']

total_data.station = pd.factorize(total_data.station)[0]

total_data.dtypes

total_data.Date = pd.to_datetime(total_data.Date, format = '%Y-%m-%d')
total_data.dtypes

lel.dtypes
lel.date =pd.to_datetime(lel.date)
lel = lel.drop(columns = 'site')
lel['site'] = site_cop
usda.dtypes
import statistics


lel['date'] = pd.to_datetime(lel['date'])
cols = lel.select_dtypes(exclude = ['datetime', 'string']).columns
print(cols)
lel[cols] = lel[cols].apply(pd.to_numeric, downcast='float', errors='coerce')

lel.dtypes
lel['site'] = pd.Series(lel['site'], dtype = 'string')

def feature_gen (total_data, usda):
    percip_feature = pd.DataFrame(columns = ['date', 'station', 'num_percip'])
    freezing_days_feature = pd.DataFrame(columns = ['date', 'station', 'num_cold'])
    warm_days_feature = pd.DataFrame(columns = ['date', 'station', 'num_warm'])
    Absolute_Percipitation = pd.DataFrame(columns = ['date', 'station', 'abs_percip'])
    avg_temp_upper = pd.DataFrame(columns = ['date', 'station', 'avg_temp_upper'])
    avg_temp_lower = pd.DataFrame(columns = ['date', 'station', 'avg_temp_lower'])
    avg_soil_temp = pd.DataFrame(columns = ['date', 'station', 'avg_soil_temp'])
    avg_rel_hum = pd.DataFrame(columns = ['date', 'station', 'avg_rel_hum'])
    avg_wind_speed = pd.DataFrame(columns = ['date', 'station', 'avg_wind_speed'])
    #total_data.Temp_avg = total_data.Temp_avg.fillna((total_data['Temp_min'] +total_data['Temp_max'])/2)
    print(1)
    usda.week_ending = pd.to_datetime(usda.week_ending)
    #compiled_df.date = pd.to_datetime(compiled_df.date)
    start = (usda.week_ending.to_numpy())
    end = (usda.week_ending - timedelta(weeks=1)).to_numpy()
    
    station_groups = total_data.groupby('site')
    print(2)
    for name, group in station_groups:
    #print(group.time)
        for i in range(0, len(start)):
            print(str(name) + '_________________________')
            st = start[i]
            en = end[i]
            print(st)
            print(en)
            temp_query = group.query('date >= @en and date <= @st')
            print(temp_query)
            Num_percip = (temp_query.precip != 0).sum()
            percip_feature = percip_feature.append({'date':start[i], 'station':name, 'num_percip':Num_percip}, ignore_index = True)
            num_cold = (temp_query.avg_air_temp <= 44).sum()
            freezing_days_feature = freezing_days_feature.append({'date':start[i], 'station':name, 'num_cold':num_cold}, ignore_index = True)
            num_warm = (temp_query.avg_air_temp >= 69).sum()
            warm_days_feature = warm_days_feature.append({'date':start[i], 'station':name, 'num_warm':num_warm}, ignore_index = True)
            abs_percip = (temp_query.precip).sum()
            Absolute_Percipitation = Absolute_Percipitation.append({'date':start[i], 'station':name, 'abs_percip':abs_percip}, ignore_index = True)
            try:            
                avg_temp_up = statistics.mean(temp_query.avg_air_temp) + (2 * statistics.stdev(temp_query.avg_air_temp))
            except:
                avg_temp_up= 0
            avg_temp_upper = avg_temp_upper.append({'date':start[i], 'station':name, 'avg_temp_upper':avg_temp_up}, ignore_index = True)
            try:            
                avg_temp_low = statistics.mean(temp_query.avg_air_temp) - (2 * statistics.stdev(temp_query.avg_air_temp))
            except:
                avg_temp_low= 0
            avg_temp_lower = avg_temp_lower.append({'date':start[i], 'station':name, 'avg_temp_lower':avg_temp_low}, ignore_index = True)
            try:            
                avg_soil_temperature = statistics.mean(temp_query.avg_soiltemp_4in_sod)
            except:
                avg_soil_temperature = 0
            avg_soil_temp = avg_soil_temp.append({'date':start[i], 'station':name, 'avg_soil_temp':avg_soil_temperature}, ignore_index = True)
            try:            
                avg_rel_humidity = statistics.mean(temp_query.avg_rel_hum)
            except:
                avg_rel_humidity = 0
            avg_rel_hum = avg_rel_hum.append({'date':start[i], 'station':name, 'avg_rel_humidity':avg_rel_humidity}, ignore_index = True)
            try:            
                avg_wind_sp = statistics.mean(temp_query.avg_wind_speed)
            except:
                avg_wind_sp = 0
            avg_wind_speed = avg_wind_speed.append({'date':start[i], 'station':name, 'avg_wind_speed':avg_wind_sp}, ignore_index = True)
            
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
        
        
    avg_soil_temp_groups = avg_soil_temp.groupby('station')
    for name, group in avg_soil_temp_groups:
        temp_col = group.avg_soil_temp.reset_index()
        Complete_Data['avg_soil_temp' + str(name) ] = temp_col.avg_soil_temp
        
         
    avg_hum_groups = avg_rel_hum.groupby('station')
    for name, group in avg_hum_groups:
        temp_col = group.avg_rel_humidity.reset_index()
        Complete_Data['avg_rel_humidity' + str(name) ] = temp_col.avg_rel_humidity

    avg_wind_speed_groups = avg_wind_speed.groupby('station')
    for name, group in avg_wind_speed_groups:
        temp_col = group.avg_wind_speed.reset_index()
        Complete_Data['avg_wind_speed' + str(name) ] = temp_col.avg_wind_speed        
        
    Complete_Data = Complete_Data.fillna(0)
    return Complete_Data

features_L = feature_gen(lel, usda)
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
    




lel = date_features(features_L, usda)



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
    
    regressor = RandomForestRegressor(n_estimators = 150, random_state = 42)
    regressor.fit(features_train, labels_train)

    predictions = regressor.predict(features_test)
    
    abs_error = abs(predictions - labels_test)
    abs_error = (np.mean(abs_error))
    print ("absolute error:" + str(abs_error))

    rmse = np.sqrt(mean_squared_error(predictions,labels_test))
    print("RMSE : % f" %(rmse))
    
   # weights = regressor.save()
    
    return predictions, abs_error, rmse, labels_test, test_dates, train_dates
    
#preds,abs_error,rmse,labels_test, test_dates, train_dates = model(data, '1980-04-27', '2018-04-08')


    
def visualizations (data, predictions, actual_values, state, commodity, test_dates):
    
    fig = plt.figure(figsize = (12,6), dpi = 80)
    ax = plt.axes()
    plt.xticks(rotation = 90)
    ax.plot(test_dates[0:8], actual_values[0:8], label = 'actual', marker = 'o')
    ax.plot(test_dates[0:8], predictions[0:8], label = 'predictions', marker = 'o')
    ax.set_ylabel('percent ' + commodity + ' planted' + ' in ' + state)
    ax.set_xlabel('date')
    plt.legend(loc = 'upper left')
    plt.show()



predictions, abs_error, rmse, labels_test, test_dates, train_dates = model(lel[11:298],'1980-01-27' , '2018-04-01')



visualizations(lel, predictions, labels_test, 'IL', 'CORN', test_dates)

plt.plot( features_IL.Date[0:20], features_IL.Percent_Planted[0:20])

features_IL['Percent_Planted'].corr(features_IL, method = 'pearson')


df = features_IL.corr(method = 'pearson')
df_spearman = features_IL_percip.corr(method = 'spearman')


dfne = NE.corr(method = 'pearson')
df_spearmanne = NE.corr(method = 'spearman')





##WEATHER.GOV API:




IL_perci1p = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/IL_percip_data1.csv')
IL_perci1p2 = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/IL_percip_data2.csv')


NE = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/NE_Corn_fulldata.csv')

IL_perci1p = IL_perci1p.append(IL_perci1p2)

features_IL_percip[20:]




def feature_gen_onlypercip (total_data, usda):
    percip_feature = pd.DataFrame(columns = ['date', 'station', 'num_percip'])
    
    Absolute_Percipitation = pd.DataFrame(columns = ['date', 'station', 'abs_percip'])


    #total_data.Temp_avg = total_data.Temp_avg.fillna((total_data['Temp_min'] +total_data['Temp_max'])/2)

    usda.week_ending = pd.to_datetime(usda.week_ending)
    
    start = (usda.week_ending.to_numpy())
    end = (usda.week_ending - timedelta(weeks=1)).to_numpy()
    station_groups = total_data.groupby('STATION')
    
    for name, group in station_groups:
    #print(group.time)
        for i in range(0, len(start)):
            print(str(name) + '_________________________')
            st = start[i]
            en = end[i]
            temp_query = group.query('DATE >= @en and DATE <= @st')
            #print(temp_query)
            Num_percip = (temp_query.PRCP != 0).sum()
            percip_feature = percip_feature.append({'date':start[i], 'station':name, 'num_percip':Num_percip}, ignore_index = True)
            abs_percip = (temp_query.PRCP).sum()
            Absolute_Percipitation = Absolute_Percipitation.append({'date':start[i], 'station':name, 'abs_percip':abs_percip}, ignore_index = True)
           
    
            i = i+1
    
    Complete_Data= pd.DataFrame()

    abs_percip_groups = Absolute_Percipitation.groupby('station')
    for name, group in abs_percip_groups:
        #print(group.abs_percip)
        temp_col = group.abs_percip.reset_index()
        Complete_Data['abs_percip_' + str(name) ] = temp_col.abs_percip
        
 
    
    
    num_percip_groups = percip_feature.groupby('station')
    for name, group in num_percip_groups:
        #print(group.num_percip)
        temp_col = group.num_percip.reset_index()
        Complete_Data['num_percip_' + str(name) ] = temp_col.num_percip
        

    Complete_Data = Complete_Data.fillna(0)
    return Complete_Data

IL_perci1p.DATE = pd.to_datetime(IL_perci1p.DATE)
percips = feature_gen_onlypercip(IL_perci1p, usda)

percips = date_features(percips, usda)


features_IL_percip = pd.merge(features_IL, percips, on ='Date')
features_IL_percip = features_IL_percip.drop(columns = ['month_y', 'week_y', 'day_y', 'Percent_Planted_y'])
features_IL_percip = features_IL_percip.drop(columns = ['abs_percip_0', 'abs_percip_1','abs_percip_2','abs_percip_3',
                                                        'abs_percip_4','abs_percip_5','abs_percip_6','abs_percip_7',
                                                        'abs_percip_8','abs_percip_9'])

features_IL_percip = features_IL_percip.rename(columns = {'Percent_Planted_x':'Percent_Planted'})



weird = IL_perci1p[IL_perci1p['STATION'] == 'USC00115712']



##Illinois state webstie data. 

loc_list = ['freday', 'frmday', 'iccday', 'llcday', 'monday', 'orrday', 'sniday']
compiled_df = pd.DataFrame()
for i in loc_list:
    dt = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/'+i+'.txt', delimiter = "\t")
    dt.to_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/'+i+ '.csv', encoding='utf-8', index=False)
    dt = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/'+i+ '.csv')

    dt = dt[['day', 'year', 'month',  'avg_wind_speed', 'max_air_temp', 'min_air_temp', 'avg_air_temp', 'avg_rel_hum', 'precip', 
             'max_soiltemp_4in_sod', 'min_soiltemp_4in_sod','avg_soiltemp_4in_sod', 'site' ]]
    dt = dt.iloc[:-10]
    dt = dt.iloc[1:]
    dt = dt.replace('----', 0)
    
    
    dt['date'] = pd.to_datetime(dt[['year', 'month', 'day']])
    dt = dt.drop(columns = ['day','year','month'])
    compiled_df = compiled_df.append(dt)

#compiled_df.to_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/UI_full_data.csv')
theta_tau = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/UI_full_data.csv')




compiled_df = compiled_df.drop(columns = ['site'])
compiled_df['site'] = lel['site']

site_cop = lel['site']

##soil features now:



