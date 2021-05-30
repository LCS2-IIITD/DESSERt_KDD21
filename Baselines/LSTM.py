# Utlity Imports
import pickle
import numpy as np
import pandas as pd
import os
import json

from tqdm import tqdm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
# %matplotlib inline

# Tensorflow and Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Parameters of Time Series Forecasting
n_steps = 20 # alpha 
n_features = 5 # Features
n_output = 1 # Future 

# time_series can be read like an array for more information look into Data_Util.zip
time_series = pickle.load(open("time_series.pkl", "rb"))

# Split the raw data into required data
def split_sequences(sequences, n_steps, n_output, start, end):
    X, y = list(), list()
    # Select all the features
    arr = sequences[0].reshape((sequences[0].shape[0], 1))
    for i in range(1, len(sequences)-1):
        arr = np.hstack((arr, sequences[i].reshape((sequences[0].shape[0], 1))))
    # Set ranges
    for i in range(start, end):
        end_ix = i + n_steps
        if end_ix + n_output >= end:
            break
        seq_x, seq_y = arr[i:end_ix, :], arr[end_ix: end_ix+n_output, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Creates the CNN Model
def getLSTMModel(X, y, n_steps=n_steps,n_output=n_output, n_features = 5):
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(n_output*n_features))
    model.compile(optimizer='adam', loss='mse')
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model.fit(X, y,  epochs=100, verbose=0,batch_size=16, callbacks=[es])
    return model

def getPredictions(X_test, model):
    test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    yhat = model.predict(test, verbose=2)
    return yhat

def createDataset(time_series, n_steps, idx, n_output):
    total_length = (time_series[idx][0].shape[0])
    splitSize = int(total_length*0.7)
    X,y = split_sequences(time_series[idx], n_steps, n_output, 0, splitSize + n_output)
    X_test,y_test = split_sequences(time_series[idx], n_steps, n_output, splitSize - n_steps - 1, total_length)
    return X, y, X_test, y_test

def run_for_index(idx, model, name, n_output, n_features):
    X, y, X_test, y_test = createDataset(time_series, n_steps, idx, n_output)
    out_shape = y.shape[1] * y.shape[2]
    y = y.reshape((y.shape[0], out_shape))
    M = model(X, y, n_steps, n_output, n_features)
    yhat = getPredictions(X_test, M)
    yhat = yhat.reshape((yhat.shape[0], n_output, n_features))
    preds = np.hstack((yhat[:-1, 0, 0], yhat[-1, :, 0]))
    orig = np.hstack((y_test[:-1, 0, 0], y_test[-1, :, 0]))
    plt.plot(preds)
    plt.plot(orig)
    # Save prediction and test by uncommenting the following line
    # pd.DataFrame((preds, orig)).to_csv(f'./{name}/{name}_{n_output}/{time_series[idx][5].strip(".csv")}_{name}_{n_output}' , header=None, index=None)

# Run for all time series
for i in tqdm(range(0,len(time_series))):
    run_for_index(i, getCNNModel, "CNN", n_output, n_features)
