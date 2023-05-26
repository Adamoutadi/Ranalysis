# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:06:58 2022

@author: ZAK0131
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.optimize import minimize
from scipy.stats import norm, t

data = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/IL_Corn_fulldata.csv')
data = data.reset_index()
data = data[data.Date >= '2011-04-10' ]

dataNE = pd.read_csv('//tedfil01/DataDropDEV/PythonPOC/Amrith/Crop_Planted_Prediction/Data/NE_Corn_fulldata.csv')


data = dataNE.drop(columns = ['Date'])

##MLE:
week_groups = data.groupby('week')

week_groups_IL = features_L[11:298].groupby('week')

features_L.week.value_counts()

##Calcualte mean and sd. Randomly sample withing one standard deviation. 

import statistics


Compiled_df = pd.DataFrame()
NE_generated_data = pd.DataFrame()

for name, group in week_groups_IL:
    sampled_data = pd.DataFrame()
    for colName, colData in group.iteritems():
        print(colData)
        mean = statistics.mean(colData.values)
        sd = statistics.stdev(colData.values)
        sample = np.random.normal(mean, sd, int(100000*(len(group)/365)))
        sampled_data[colName] = sample
        sampled_data['week'] = name
    Compiled_df = Compiled_df.append(sampled_data)
    
    
data = data.set_index(data.Date)
data = data.drop(columns = ['Date'])

dd = week_groups.get_group(16)

##WRONG GOTTA FIX
import numpy as np
for name, group in week_groups:
    print(name)
    test = pd.DataFrame()
    for colName, colData in group.iteritems():
        print(colName)
        print(colData.values)
        mean = statistics.mean(colData.values)
        print(mean)
        sd = np.std(colData.values, ddof = 1)
        print(sd)
        sample = np.random.normal(mean, sd, 100000)
        test[colName] = sample
    
    NE_generated_data = NE_generated_data.append(test)
    
NE_generated_data = NE_generated_data.dropna()
    
    
    