# Importing the modules 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# loading the data
dataset_train = pd.read_csv('aapl.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

Creating Data with Timesteps

 
"""LSTMs expect our data to be in a specific format, usually a 3D array. We start by creating data in 60 timesteps and converting it
into an array using NumPy. Next, we convert the data into a 3D dimension array with X_train samples, 60 timestamps,
and one feature at each step."""

X_train = []
y_train = []
for i in range(60, 400):
    X_train.append(training_set_scaled[i-60 : i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building The LTSM

"""When defining the Dropout layers, we specify 0.2, meaning that 20% of the layers will be dropped. Thereafter,
we add the Dense layer that specifies the output of 1 unit. After this, we compile our model using the popular adam optimizer 
and set the loss as the mean_squarred_error. This will compute the mean of the squared errors. Next, we fit the model to run on 
100 epochs with a batch size of 32. Keep in mind that, depending on the specs of your computer, this might take a few minutes to
finish running."""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Predicting the future stock using test set

dataset_test = pd.read_csv('aapl.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Plotting the Results

plt.plot(real_stock_price, color = 'black', label = 'TATA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()
