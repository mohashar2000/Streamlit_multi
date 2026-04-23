# 📦 STEP 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import yfinance as yf
from datetime import datetime

# 📈 STEP 2: Load Stock Data (Apple - AAPL)
end_date = datetime.now()
start_date = datetime(end_date.year - 1, end_date.month, end_date.day)
data = yf.download("AAPL", start=start_date, end=end_date)

# 👀 STEP 3: Explore the Data
print(data.head())
fig = px.line(data, x=data.index, y="Close", title="Apple Stock Closing Price")
fig.show()

# 🧮 STEP 4: Prepare Data
df_close = data[["Close"]]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_close)

# Create training data (past 60 days → predict next day)
x_train, y_train = [], []
for i in range(60, len(scaled_data)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 🧠 STEP 5: Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# 🏋️ STEP 6: Train the Model
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, batch_size=1, epochs=3)

# 🔮 STEP 7: Make Predictions
train_len = int(len(scaled_data) * 0.95)
test_data = scaled_data[train_len - 60:]
x_test = []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# 📊 STEP 8: Visualize Predictions
train = df_close[:train_len]
valid = df_close[train_len:]
valid["Predictions"] = predictions

plt.figure(figsize=(12,6))
plt.plot(train["Close"], label="Training Data")
plt.plot(valid[["Close", "Predictions"]])
plt.title("Apple Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend(["Train", "Actual", "Predicted"])
plt.show()
