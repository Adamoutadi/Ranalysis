# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:28:43 2022

@author: ZAK0131
"""

import pandas as pd
import numpy as np
print(pd.__version__)


import pickle as pkl
import pandas as pd
with open("//tedfil01/DataDropDEV/PythonPOC/CQG/usdadata.pkl", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv("//tedfil01/DataDropDEV/PythonPOC/Upload_CSVs/usdaData.csv")

df = pd.read_csv("//tedfil01/DataDropDEV/PythonPOC/Upload_CSVs/usdaData.csv")
df

import os
os.chdir('//tedfil01/DataDropDEV/PythonPOC/')
#import dbUtils as db
import pyodbc
import dbUtils as db


SWE = db.sqlToDataFrame('RiskPOC',"""SELECT  [dt] Dates,[Basin]+ ' - ' + Variable LocAndMeasure
      ,[SumMeasure] Reading
  FROM [RiskPOC].[dbo].[Snodas]
  where variable in ('SWE','Liquid precipitation')
   UNION
  SELECT [dt] Dates,[Basin]+ ' - ' + Variable LocAndMeasure
  ,[SumMeasure] Reading
  FROM [RiskPOC].[dbo].[DailySnodas]
  order by dt""")


##Machine Learning test:
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

df.describe().transpose()

'''
date_time = pd.to_datetime(df.pop('dt'), format = '%Y-%m-%d')
time_stamps = date_time.map(pd.Timestamp.timestamp)

day = 24 * 60 * 60
year = (365.2425) * day
df['Day sin'] = np.sin(time_stamps * (2 * np.pi / day))
df['Day cos'] = np.cos(time_stamps * (2 * np.pi / day))
df['Year sin'] = np.sin(time_stamps * (2 * np.pi / year))
df['Year cos'] = np.cos(time_stamps * (2 * np.pi / year))
'''
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


#train val test split
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]
#Normalization
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

#significance tests:
from scipy import stats
spearman_df = pd.DataFrame(columns = ['name', 'coefficient'])
for i in df.columns:
    spearman_df = spearman_df.append({'name':i, 'coefficient':stats.spearmanr(df.Planted, df[i])[0]}, ignore_index = True)
    
## Existing weather data is not linearly or non-linearlly related

#window generator for time series prediction
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

    
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window


def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)

  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset


@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

null_looker = df.isnull().any()

Num_epochs = 25

train_df = train_df.dropna(axis = 1)
val_df = val_df.dropna(axis = 1)
test_df = test_df.dropna(axis = 1)

def compile_and_fit(model, window, patience=2):


  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=Num_epochs,
                      validation_data=window.val)
  return history

#linear model. In keras a dense layer with no activation function is linear
single_step_window = WindowGenerator(input_width =10, label_width = 10, shift =1, label_columns = ['Planted'])
for example_inputs, example_labels in single_step_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

print('Input shape:', single_step_window.example[0].shape)
#print('Output shape:', linear(single_step_window.example[0]).shape)
print(tf.__version__)


linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])
history = compile_and_fit(linear, single_step_window)

#Dense model:
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(dense, single_step_window)







'''
df_copy = df
df  = df.drop(['Unnamed: 0','dt'], axis =1)

values = df.values




values = values.astype('float32')
#normalizes data
scaler = MinMaxScaler(feature_range = (0,1))
values = scaler.fit_transform(values)
       

#train/test sets
                
train_size = int(len(values)* 0.8) 

test_size = len(values) - train_size
train, test = values[0:train_size, :], values[train_size:len(values), :]

train_X, train_y = train[:, 1:], train[:, 0]
test_X, test_y = test[:, 1:], test[:, 0]
# gives x - datatrain.iloc[:, 1:]
#train.iloc[:, 1] gives y- train \data



#reshaping into [samples, timesteps, faetures] form that LSTM expects
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))


#train_y.shape

#LSTM model
model = Sequential()


#model.add(LSTM(80, activation='relu', return_sequences = True, input_shape = (train_X.shape[1], train_X.shape[2])))

model.add(LSTM(120, return_sequences = True, input_shape = (train_X.shape[1],train_X.shape[2]), activation = 'relu')) #gives 50 neurons in the first hidden layer
#model.add(LSTM(50, activation = 'tanh'))
#model.add(LSTM(50, activation='relu'))
#model.add(Dense(5)) #1 neuron in output layer
#opt = tf.keras.optimizers.Adam(0.001)
model.add(Dropout(0.2))
#model.add(LSTM(120,activation = 'relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

#BEST: 50 neurons in 1 layer with dorpout of 0.4, tanh activation, epochs - 5, batch size - 128 -- 35 RMSE


#fitting
history = model.fit(train_X, train_y, epochs =25, batch_size = 64, validation_data = (test_X, test_y), verbose = 2, shuffle = False)


#plotting
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'test')

plt.legend()
plt.ylim([0,0.5])

plt.show()


#prediction
predicted_y = model.predict(test_X)



#Uncommonent and index predicted_values to get the predicted y-values
#test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#inverse_predicted_y = np.concatenate((predicted_y, test_X), axis = 1)
#
#predicted_values = inverse_predicted_y
#
#
#predicted_values = scaler.inverse_transform(predicted_values)

#predicted_values[-1,0]


print(predicted_y)
print(predicted_y.shape)

predicted_y = predicted_y.reshape((predicted_y.shape[0],predicted_y.shape[2]))
print(predicted_y.shape)
#test_test_y = scaler.inverse_transform(predicted_y)

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))


#inversion of predicted
inverse_predicted_y = np.concatenate((predicted_y, test_X), axis = 1) #test_X indexing may be wrong


testing_y_values= scaler.inverse_transform(inverse_predicted_y)# -- to get the unscaled RMSE


inverse_predicted_y = inverse_predicted_y[:,0]
print(inverse_predicted_y)


#actual inverseion
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X), axis =1)
#inv_y = scaler.inverse_transform(inv_y) --- to get the unscaled RMSE
inv_y = inv_y[:,0]

rmse = sqrt(mean_squared_error(inv_y, inverse_predicted_y)) # compute it on the scaled data NOT THE unscaled 

print('RMSE value:',rmse)

'''




