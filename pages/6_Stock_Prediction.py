
# LSTM : STock Prediction

# What was the change in price of the stock over time ?
# What was the daily return of the stock on average ?
# What was the moving average of the various stocks ?
# What was teh correlation between different stocks?
# How much value do we put at risk by investing in a particular stock?
# How can we attempt to predict future stock behavious? (Predicting the closing price stock price of APPLE inc using LSTM)
# https://pypi.org/project/yfinance/

#Stock API
# pip install yfinance pandas numpy matplotlib seaborn plotly scikit-learn tensorflow keras
# pip install tensorflow keras scikit-learn
# pip install tensorflow


#  Step 1 - import the libraries
# Configure the libraries
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

#common libraries
try:
    import numpy as np
    import pandas as pd
    #import seaborn as sns
    import plotly.express as px
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install required packages: pip install pandas numpy matplotlib seaborn plotly")
    exit(1)

#Machine Learning libraries
try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError as e:
    print(f"Missing ML library: {e}")
    print("Please install: pip install scikit-learn")
    MinMaxScaler = None

# TensorFlow/Keras imports (install with: pip install tensorflow)
from keras.layers import LSTM, Dense
from keras.models import Sequential
# try:
#     from tensorflow.keras.utils import TimeseriesGenerator
# except ImportError:
#     from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import os
import tensorflow as tf
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"



#import yfinance
try:
    import yfinance as yf
except ImportError as e:
    print(f"Missing yfinance library: {e}")
    print("Please install: pip install yfinance")
    exit(1)

# Grabbing the data from YFinance
data = yf.Ticker("AAPL")

hist = data.history(period = '1mo')

print(data.info)

#converting the data into datetime format
current_data = datetime.now()   # This code will help you to get the current date

print(current_data)

start_date = datetime(current_data.year -1, current_data.month, current_data.day -1)

print(start_date)

data = yf.download("AAPL",start_date, current_data)

print(data.head())

print(data.tail())

#Visualize the data for understanding the crucial part of the data\\\
    
fig = px.line(data["Close"],title="Apple Stock Data")
fig.show()

print(data.info())
# # Data description
print(data.describe())

# # Plot the distribution for open columns (requires seaborn)
# sns.distplot(data["Close"])
# Uncomment above line after installing seaborn: pip install seaborn

# This disribution tells us the event after having good amount of data points being scattered around teh mean and median were skill skewed due to 
# few bigger points 

# #Step2 - Information on Closing Prices**


# #visualize Closing Prices

plt.figure(figsize =(15,6))
plt.plot(data.Close, color="red")
plt.title("Close price")
plt.xlabel("Date of Stock")
plt.ylabel("Closing Price")
plt.show()



# # Qn1. What was the change in price for the stock Over time ?**

# # Yes . as it can be seen in the above graph the prices of stock is completely different
# # from what we have seen in the last year around the same time

# **Step 3 - Infomration on the volume of the assets sold **

# # It is the number/ quanity of assets sold or taded between daily open and close
# #visualize Closing Prices

plt.figure(figsize =(15,6))
plt.plot(data.Volume, color="blue")
plt.title("Volume of Stock")
plt.xlabel("Date of Stock")
plt.ylabel("Volume")
plt.show()

# # The above graph shows, we can conclude that the trade volume was normal 
# # intial period and April huge inflation and in October.
# # In the month of April, there were a lot of trading done as compared to other months

# # **Step 4  - Working with Moving average **



# # This will help us to identify the udpate that wer doen according to the specific time frame

# # fine the Moving average for the given data for different time frame

moving_average = [ 10, 20, 50]
for x in moving_average:
  columns_name =f"MA for {x} days"
  data[columns_name] = data ["Close"].rolling(x).mean()
  
  data = data.reset_index()   # make Date a column

# # Visulazing to udnerstand the moving average data

print(data.head())

# # If columns are multi-index, flatten them
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)  # keep only first level

# # Reset index to have Date as a column
#data = data.reset_index()

#print(data.head())

data['MA for 10 days'] = data['Close'].rolling(window=10).mean()
data['MA for 20 days'] = data['Close'].rolling(window=20).mean()
data['MA for 50 days'] = data['Close'].rolling(window=50).mean()

fig = px.line(data, x=data.index, y= ["Close", "MA for 10 days", "MA for 20 days", "MA for 50 days"])
fig.show()

data["Daily Return"] = data["Close"].pct_change() 
print(data.head())

fig = px.line(data, x=data.index, y="Daily Return", title="Change in the stock on daily basis")   
fig.show()

fig = px.histogram(data, x="Daily Return", title ="Change in stock pricing %")
fig.show()

# # If we are investing in this stock, there is a faily high chance that the chagnes **are going between 0.05 to -0.05 bold text

# #Double-click (or enter) to edit

# #**Step 6 - Training and Testing Data **

# Ashraf
data = yf.download("AAPL", start = "2022-08-15", end = datetime.now())

#print(data.head())

# # If columns are multi-index, flatten them
if isinstance(data.columns, pd.MultiIndex):
   data.columns = data.columns.get_level_values(0)  # keep only first level
print(data.head())

fig = px.line(data, x=data.index, y="Close", title="Apple Stock Data")
fig.show()

#Create a Seperate dataframe that will only have target column for predictions
df_Close = data["Close"]

# #Ashraf
df_Close = df_Close.to_frame()

# # will try to keep around 95% of dat afor training purpose
train_len = int(np.ceil(len(df_Close)*0.95))

print(train_len)



# # Step 7 - Building the data for prediction

# # Scaling the values to remove the ahead bias from the data

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_Close.values.reshape(-1, 1))

pd.DataFrame(data_scaled, columns = ["Scaled Columns"] )

# # Step 8 - Split the data into training and testing data
# # need to create the training data
train_data = data_scaled[0 : train_len, :]

# wil will try to split this training data into x and y
x_train, y_train = [], []

# we will try to create a sequence of data, where we will use a series of past valuees to predict the upcoming future values

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])
  
  #conver the give daa into numpy array for the model 6
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

# Working with LSTM network (requires TensorFlow installation)
# Uncomment the following code after installing TensorFlow: pip install tensorflow

model = Sequential()
model.add(LSTM(128, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(64, return_sequences = False))
model.add(Dense(30))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, batch_size =1, epochs=2)

#Test data creation
test_data = data_scaled[train_len - 60:,:]
x_test = []
y_test = df_Close.values[train_len : , :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

#Predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Evaluation
RMSE = np.sqrt(np.mean((y_test - predictions)**2))
print(f"RMSE :{RMSE}")

#Visualize predictions
train = df_Close[:train_len]
prediction_data= df_Close[train_len:]
prediction_data["Predictions"] = predictions
a = prediction_data.Close.values
b = prediction_data.Predictions
fig = px.line(train, x=train.index, y="Close", title = "Apple Stock Price Prediction using LSTM")
fig.add_scatter(x = prediction_data.index, y=a, name = "Actual data")
fig.add_scatter(x = prediction_data.index, y=b, name = "Prediction data")
fig.show()

print("LSTM model code is commented out. Install TensorFlow to enable: pip install tensorflow")

