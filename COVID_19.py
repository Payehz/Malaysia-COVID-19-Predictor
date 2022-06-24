# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:44:31 2022

@author: User
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard


from modules_for_covid import EDA, ModelCreation, Results

#%% Statics

CSV_TRAIN_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
MMS_PATH = os.path.join(os.getcwd(),'saved_models','mms.pkl')
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'covid_logs', log_dir)
CSV_TEST_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','model.h5')

#%% Step 1) Data Loading

df = pd.read_csv(CSV_TRAIN_PATH)

#%% Step 2) Data Inspection

df.info()
df.describe().T

# There are some empty values in the data

# Plotting only the cases_new feature
eda = EDA()
eda.plot_graph(df) # The graph looks weird due to the empty spaces in the data


#%% Step 3) Data Cleaning

# Converting empty data to NaN
df['cases_new'] = pd.to_numeric(df['cases_new'],errors='coerce')
df.info()

# Interpolating NaNs to fit the data/graph
df['cases_new'] = df['cases_new'].interpolate()
df.isna().sum()

eda.plot_graph(df) # Now it looks better

#%% Step 4) Features Selection

# We are selecting only the cases_new feature

#%% Step 5) Data Preprocessing

# Use MinMaxScaler to scale the data
mms = MinMaxScaler()
df = mms.fit_transform(np.expand_dims(df['cases_new'],axis=-1))

with open(MMS_PATH,'wb') as file:
    pickle.dump(mms,file)

X_train = []
y_train = []

win_size = 30

for i in range(win_size,np.shape(df)[0]):
    X_train.append(df[i-win_size:i,0])
    y_train.append(df[i,0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)

#%% Model Development

num_feature = np.shape(X_train)[0]

# Train the model
mc = ModelCreation()
model = mc.two_layer_model(num_feature)

tensorboard_callback = TensorBoard(log_dir=LOG_FOLDER_PATH)

model.compile(optimizer='adam',loss='mse',metrics='mape')

X_train = np.expand_dims(X_train,axis=-1)
hist = model.fit(X_train,y_train,batch_size=64,epochs=100,
                 callbacks=[tensorboard_callback])

model.save(MODEL_SAVE_PATH)

#%% Plot model

plot_model(model,show_shapes=True,show_layer_names=(True))

#%% Model Evaluation

hist.history.keys()

# Loss graph plot
mc.loss_graph(hist)

#%% Model development and analysis

# Load test data
test_df = pd.read_csv(CSV_TEST_PATH)

# Fill the NaN using interpolate
test_df['cases_new'] = test_df['cases_new'].interpolate()

# Scale it with MinMaxScaler
test_df = mms.transform(np.expand_dims(test_df['cases_new'],axis=-1))

# Concatenate the test data at the end of train data
con_test = np.concatenate((df,test_df),axis=0)

con_test = con_test[-130:]

X_test =[]
for i in range(win_size,len(con_test)):
    X_test.append(con_test[i-win_size:i,0])
    
X_test = np.array(X_test)

predicted = model.predict(np.expand_dims(X_test,axis=-1))

#%% Plotting of graphs

results = Results()

# Getting the actual and predicted graph
results.result_graph(test_df, predicted)

#%% MSE, MAE, MAPE

# Getting the MSE, MAE and MAPE
results.loss_scores(test_df, predicted)

#%% Discussion

# This model can predict the trend of the COVID-19 cases.
# Despite error is around mean absolute error of 6% and error is only
# around 14% for MAPE when tested against testing dataset.



