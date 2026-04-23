import streamlit as st


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
# from keras.layers import LSTM, Dense
# from keras.models import Sequential
# try:
#     from tensorflow.keras.utils import TimeseriesGenerator
# except ImportError:
#     from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator



#import yfinance
try:
    import yfinance as yf
except ImportError as e:
    print(f"Missing yfinance library: {e}")
    print("Please install: pip install yfinance")
    exit(1)

# Streamlit app title
st.title("Dropdown Selection Example")

# Dropdown options
options = ["CHOOSE Ticker to get the trade prediction","APPLE", "META", "TESLA", "GOOGLE"]

# Dropdown box
ticker = st.selectbox("Select your favorite fruit:", options)

ticker_symbol = ""

    
match ticker:
    case "CHOOSE Ticker to get the trade prediction":
        ticker_symbol = "You did not select any ticket. Please select the ticker"
    case "META":
        ticker_symbol = "META"
    case "TESLA":
        ticker_symbol = "TSLA"
    case "GOOGLE":
        ticker_symbol = "GOOGL"
    case "APPLE":
        ticker_symbol = "APPL"
    


# Display selected value
st.write(f"You selected: **{ticker_symbol}**")

#converting the data into datetime format
current_data = datetime.now()   # This code will help you to get the current date

print(current_data)

start_date = datetime(current_data.year -1, current_data.month, current_data.day -1)

print(start_date)

data = yf.download(ticker_symbol,start_date, current_data)

print(data.head())

print(data.tail())

moving_average = [ 10, 20, 50]
for x in moving_average:
  columns_name =f"MA for {x} days"
  data[columns_name] = data ["Close"].rolling(x).mean()
  
  data = data.reset_index()   # make Date a column
  
#   # # If columns are multi-index, flatten them
# if isinstance(data.columns, pd.MultiIndex):
#    data.columns = data.columns.get_level_values(0)  # keep only first level
# print(data.head())


# data['MA for 10 days'] = data['Close'].rolling(window=10).mean()
# data['MA for 20 days'] = data['Close'].rolling(window=20).mean()
# data['MA for 50 days'] = data['Close'].rolling(window=50).mean()

# #Create a Seperate dataframe that will only have target column for predictions
# df_Close = data["Close"]

# # #Ashraf
# df_Close = df_Close.to_frame()

# # # will try to keep around 95% of dat afor training purpose
# train_len = int(np.ceil(len(df_Close)*0.95))

# print(train_len)



# # # Step 7 - Building the data for prediction

# # # Scaling the values to remove the ahead bias from the data

# scaler = MinMaxScaler()
# data_scaled = scaler.fit_transform(df_Close.values.reshape(-1, 1))

# pd.DataFrame(data_scaled, columns = ["Scaled Columns"] )

# # # Step 8 - Split the data into training and testing data
# # # need to create the training data
# train_data = data_scaled[0 : train_len, :]

# # wil will try to split this training data into x and y
# x_train, y_train = [], []

# # we will try to create a sequence of data, where we will use a series of past valuees to predict the upcoming future values

# for i in range(60, len(train_data)):
#   x_train.append(train_data[i-60:i,0])
#   y_train.append(train_data[i,0])
  
#   #conver the give daa into numpy array for the model 6
# x_train, y_train = np.array(x_train), np.array(y_train)

# #Reshape the array
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
