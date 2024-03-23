import os
#third party packages I'll be using
import math
import pandas_datareader as web
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime as dt
plt.style.use("fivethirtyeight")
#This project will use an aritifcial neural network model called LSTM(Long short term memory) model which will be used to predict the closing stock price of a company(Apple Inc) with 60 days of past performance
#Limitations of Project that I would like to addreess in the future:
#Caching HTTP requests so instead of making new requests I can reuse old ones that are cached
#Limiting the amount of HTTP requests I make so as not to get blocked by the website Im requesting from basically
# from requests import Session
# from requests_cache import CacheMixin, SQLiteCache
# from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
# from pyrate_limiter import Duration, RequestRate, Limiter
#ChatGPT4 suggestions for improving the horrid memory consumption of this project
# Memory Optimization: Use more memory-efficient data structures or libraries where possible. For example, consider using generators or streaming data processing techniques to avoid loading the entire dataset into memory at once.
# Batch Processing: If training or prediction operations consume too much memory, try processing data in smaller batches rather than all at once.
# Model Complexity: Simplify the neural network architecture or reduce the number of parameters in the model to lower memory usage.
# Plotting Optimization: Optimize plotting code to reduce memory usage, such as using smaller plot sizes or limiting the number of plotted data points.
# class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
#     pass

# session = CachedLimiterSession(
#     limiter=Limiter(RequestRate(2, Duration.SECOND*5)),  # max 2 requests per 5 seconds
#     bucket_class=MemoryQueueBucket,
#     backend=SQLiteCache("yfinance.cache"),
# )

#Getting the stock quote
appl_start = dt.datetime(2012, 1, 31)
appl_end = dt.datetime(2024, 1, 31)

df = yf.download('AAPL', start=appl_start, end=appl_end)
dataframe = pd.DataFrame(df)
#print(df)
plt.figure(figsize=(16,8))
plt.title("Closing Price History")
plt.plot(dataframe["Close"])
plt.xlabel("Date")
plt.ylabel("United States Dollar")
#plt.show()


#Create a new dataframe with only the close column
data = dataframe.filter(["Close"])
dataset = data.values
training_data_length = math.ceil(len(dataset) * 0.8)

#Scaling Data, apply preprocessing transformations/scaling/normalizationt to input data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create the trainig dataset
#Create the scaled training dataset
train_data = scaled_data[0:training_data_length]
#Split the data into X_train and Y_train
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0]) #X train contains 60 values from 0, 59
    y_train.append(train_data[i, 0])      #Y train will contain the 61st value at position 60
    if(i <= 60):
        print(x_train)
        print(y_train)
        print()

#Convert x_train and y_train into numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the x_train dataset, LSTM expects the data to be 3 Dimensional but our data is 2 Dimensional so we have to format it into a 3Dimensional model
print(x_train.shape)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

#Compiling the model
#Train the model
model.compile(optimizer = "adam", loss="mean_squared_error")
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Create the testing data
test_data = scaled_data[training_data_length - 60:, :]

#create the test data sets
x_test = []
y_test = dataset[training_data_length:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60: i, 0])
x_test = np.array(x_test)   
print(x_test.shape)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the models predicted price value for x_test dataset
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) #Makes predictions contain the same values as our y_test dataset, getting predictions based off of x_test dataset

#Get the root mean square error, standard deviation of residuals
rmse = np.sqrt(np.mean(predictions-y_test)**2)

train = data[:training_data_length]
valid = data[training_data_length:]
valid["Predictions"] = predictions
plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel("Date")
plt.ylabel("Close Price USD($) Value")
#print(f"This is what the valid value equates to {valid}")
#print(f"This is the value of valid[predictions] {valid[predictions]}")
plt.plot(train["Close"])
plt.plot(valid[["Close", "Predictions"]])
plt.legend(["Train", "Val", "Predictions"], loc="lower right")
plt.show()

#Show valid price against predicted price
