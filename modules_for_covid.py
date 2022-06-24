# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:03:01 2022

@author: User
"""

import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error


#%% Statics

MMS_PATH = os.path.join(os.getcwd(),'saved_models','mms.pkl')

#%% Load models

with open(MMS_PATH,'rb') as file:
    mms = pickle.load(file)

#%% Classes

class EDA():
    def __init__(self):
        pass
    
    def plot_graph(self,df):
        
        plt.figure()
        plt.plot(df['cases_new'])
        plt.legend(['cases_new'])
        plt.show()

class ModelCreation():
    def __init__(self):
        pass
    
    def two_layer_model(self,num_feature,drop_rate=0.2,num_nodes=64):
        
        model = Sequential()
        model.add(Input(shape=(num_feature,1)))
        model.add(LSTM(num_nodes,return_sequences=True))
        model.add(Dropout(drop_rate))
        model.add(LSTM(num_nodes))
        model.add(Dropout(drop_rate))
        model.add(Dense(128))
        model.add(Dropout(drop_rate))
        model.add(Dense(1))
        model.summary()
        
        return model
    
    def loss_graph(self,hist):
        
        plt.figure()
        plt.plot(hist.history['mape'])
        plt.plot()

        plt.figure()
        plt.plot(hist.history['loss'])
        plt.plot()
    
class Results():
    def __init__(self):
        pass
    
    def result_graph(self,test_df,predicted):
        
        plt.figure()
        plt.plot(test_df,'b',label='actual COVID cases')
        plt.plot(predicted,'r',label='predicted COVID cases')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(mms.inverse_transform(test_df),'b',label='actual COVID cases')
        plt.plot(mms.inverse_transform(predicted),'r',label='predicted COVID cases')
        plt.legend()
        plt.show()
        
    def loss_scores(self,test_df,predicted):
        
        print("mse: " + str(mean_squared_error(test_df,predicted)))
        print("mae: " + str(mean_absolute_error(test_df,predicted)))
        print("mape: " + str(mean_absolute_percentage_error(test_df,predicted)))

        test_df_inversed = mms.inverse_transform(test_df)
        predicted_inversed = mms.inverse_transform(predicted)

        print("inversed mse: " + str(mean_squared_error(test_df_inversed,predicted_inversed)))
        print("inversed mae: " + str(mean_absolute_error(test_df_inversed,predicted_inversed)))
        print("inversed mape: " + str(mean_absolute_percentage_error(test_df_inversed,predicted_inversed)))



