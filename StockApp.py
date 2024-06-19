import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def predict_future_prices(model, input_data, num_days):
    predictions = []
    last_sequence = input_data[-model.input.shape[1]:].reshape(1, -1, 1)
    
    for _ in range(num_days):
        prediction = model.predict(last_sequence)
        predictions.append(prediction[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1, -1] = prediction
    
    return predictions

st.title('Stock Price Prediction with Moving Averages and RSI')

user_input = st.text_input('Enter Stock Ticker', 'SBIN.NS')
stock = yf.Ticker(user_input)
data = stock.history(period="10y", interval="1d")
df = pd.DataFrame(data)

st.subheader('Data from past 10 years')
st.write(df.describe())

st.subheader('Closing Price Vs Time')
fig_price = plt.figure(figsize=(15, 7))
plt.plot(df.Close, 'b', label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig_price)

st.subheader('Moving Averages and Relative Strength Index(RSI)')

# Sliders for SMA and RSI window lengths
sma_window_1 = st.slider('Select first SMA window', min_value=2, max_value=100, value=60, step=1)
sma_window_2 = st.slider('Select second SMA window', min_value=2, max_value=200, value=120, step=1)
rsi_window = st.slider('Select RSI window', min_value=2, max_value=50, value=14, step=1)

mvavg1 = df.Close.rolling(sma_window_1).mean()
mvavg2 = df.Close.rolling(sma_window_2).mean()
rsi = calculate_rsi(df.Close, window=rsi_window)

fig_sma_rsi = plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(mvavg1, label=f'{sma_window_1} Days SMA')
plt.plot(mvavg2, label=f'{sma_window_2} Days SMA')
plt.plot(df.Close, 'b', label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(rsi, 'g', label=f'RSI ({rsi_window} Days)')
plt.xlabel('Time')
plt.ylabel('RSI')
plt.legend()

st.pyplot(fig_sma_rsi)

train_data = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
test_data = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))

train_data_array = scaler.fit_transform(train_data)
test_data_array = scaler.transform(test_data)

x_train = []
y_train = []

for i in range(60, train_data_array.shape[0]):
    x_train.append(train_data_array[i-60:i])
    y_train.append(train_data_array[i, 0])   

x_train, y_train = np.array(x_train), np.array(y_train)

model = load_model('keras_model.h5')

days30 = train_data.tail(60)
final_df = days30._append(test_data, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(60,input_data.shape[0]):
    x_test.append(input_data[i-60: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_pred = y_pred*scale_factor
y_test = y_test*scale_factor

st.subheader("Original_Values & Predicted_Values Vs Time")
fig = plt.figure(figsize = (20,10))
plt.plot(y_test,'b',label='Original_Value')
plt.plot(y_pred,'r',label='Predicted_value')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


