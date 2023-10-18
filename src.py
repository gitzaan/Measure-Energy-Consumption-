import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.preprocessing
from sklearn.metrics import r2_score
from keras.layers import Dense, Dropout, SimpleRNN, LSTM
from keras.models import Sequential

# Load AEP_hourly.csv
aep_data = pd.read_csv('AEP_hourly.csv', parse_dates=['Datetime'], index_col='Datetime')

# Data Cleaning
aep_data_cleaned = aep_data.dropna()

# Data Splitting
from sklearn.model_selection import train_test_split

aep_X = aep_data_cleaned.drop(columns=['AEP_MW'])
aep_y = aep_data_cleaned['AEP_MW']

aep_X_train, aep_X_test, aep_y_train, aep_y_test = train_test_split(aep_X, aep_y, test_size=0.2, random_state=42)

# Data Normalization
from sklearn.preprocessing import MinMaxScaler

aep_scaler = MinMaxScaler()
aep_y_train_normalized = aep_scaler.fit_transform(aep_y_train.values.reshape(-1, 1))
aep_y_test_normalized = aep_scaler.transform(aep_y_test.values.reshape(-1, 1))

# Data Reshaping
sequence_length = 24

def create_sequences(X, y, sequence_length):
    X_sequences, y_sequences = [], []
    for i in range(len(X) - sequence_length):
        X_sequences.append(X.iloc[i:i+sequence_length].values)
        y_sequences.append(y[i+sequence_length])
    return np.array(X_sequences), np.array(y_sequences)

aep_X_train_lstm, aep_y_train_lstm = create_sequences(aep_X_train, aep_y_train_normalized, sequence_length)
aep_X_test_lstm, aep_y_test_lstm = create_sequences(aep_X_test, aep_y_test_normalized, sequence_length)

# Conclusion
print("Phase 3 completed successfully.")
