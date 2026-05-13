import pickle
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

print("Loading LSTM Weights...")
try:
    actions = np.array(['hello', 'thanks', 'iloveyou'])
    model_lstm = Sequential()
    model_lstm.add(Input(shape=(30, 1662)))
    model_lstm.add(LSTM(64, return_sequences=True, activation='relu'))
    model_lstm.add(LSTM(128, return_sequences=True, activation='relu'))
    model_lstm.add(LSTM(64, return_sequences=False, activation='relu'))
    model_lstm.add(Dense(64, activation='relu'))
    model_lstm.add(Dense(32, activation='relu'))
    model_lstm.add(Dense(actions.shape[0], activation='softmax'))
    
    model_lstm.load_weights('action.h5')
    print("LSTM Weights Loaded Successfully!")
except Exception as e:
    print("Error loading LSTM Weights:", e)
