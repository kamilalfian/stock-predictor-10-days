import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import yfinance as yfin
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import keras.backend as K
import tensorflow as tf
from matplotlib.lines import Line2D

#Title for the web app
st.title('Stock Prediction')

#Fetching stock data
user_input=st.text_input('Enter Stock Ticker (based on Yahoo Finance Website, ex: INTP.JK)','INTP.JK')
start='2013-01-01'
end=datetime.now().strftime('%Y-%m-%d')
yfin.pdr_override()
df=yfin.download(user_input,start,end)

#Subtitle
st.subheader('Data from 2013-Today')

#Various data description including chart with MA
st.write(df.sort_index(ascending=False))
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with MA100 and MA200')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close, 'b',label='Closing Price',)
plt.plot(ma100,'g',label='MA100')
plt.plot(ma200,'r',label='MA200')
plt.legend()
st.pyplot(fig)

#splitting data into training/test
train=pd.DataFrame(df.Close[0:int(len(df)*0.7)])
test=pd.DataFrame(df.Close[int(len(df)*0.7):int(len(df))])

#scaling data to prevent vanishing gradient and make it compatible for deep learning
scaler=MinMaxScaler()
train=scaler.fit_transform(train)

#preparing the same custom loss function being used in the pretrained model
@tf.function
def custom_loss(y_test, y_predicted):
    # Assuming y_true and y_pred are tensors with arbitrary dimensions

    # Flatten the tensors to 1D arrays
    y_true_flat = K.flatten(y_test)
    y_pred_flat = K.flatten(y_predicted)

    # Assign higher weight to more recent data
    weights = K.arange(1, K.shape(y_true_flat)[0] + 1, dtype='float32')
    weights = K.reverse(weights, axes=0)
    weights /= K.sum(weights)

    # Calculate mean squared error with weighted samples
    loss = K.mean(weights * K.square(y_true_flat - y_pred_flat), axis=-1)

    return loss

# Register the custom loss function
tf.keras.utils.get_custom_objects()['custom_loss'] = custom_loss

# Load pretrained model, created by Stock_Predictor.ipynb
model = load_model('model.h5')

# Compile the model with the custom loss function
model.compile(optimizer='adam', loss='custom_loss')

#load testing data
past100=pd.DataFrame(df.Close[0:int(len(df)*0.7)]).tail(100)
final_df=past100.append(test,ignore_index=False)
input_data=scaler.transform(final_df)
x_test=[]
y_test=[]
for i in range(100, input_data.shape[0]+1):
    x_test.append(input_data[i-100:i])
for i in range(100, input_data.shape[0]-9):
    y_test.append(input_data[i:i+10,0])
x_test, y_test = np.array(x_test), np.array(y_test)

#predict the stock price using the testing data
y_predicted = model.predict(x_test)

#revert scaling to give actual price
y_predicted=(y_predicted*(scaler.data_max_-scaler.data_min_)+scaler.data_min_)
y_test=y_test*(scaler.data_max_-scaler.data_min_)+scaler.data_min_

# Calculate MAPE
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred[:-10]) / y_true)) * 100

# Calculate MAPE for your predictions
mape = calculate_mape(y_test, y_predicted)

# Write the tabular data for the original price and predicted price
y_test=pd.DataFrame(y_test,index=test.index[:-9],columns=[f'Original Price ({date.strftime("%d/%m/%Y")})' for date in [test.index[-9] + pd.offsets.BDay(i) for i in range(1, 11)]])
y_predicted = pd.concat([pd.DataFrame(y_predicted[:-1], index=test.index, columns=[f'Predicted Price (D+{i})' for i in range(0, 10)]),
                         pd.DataFrame(y_predicted[-1:], index=[test.index[-1] + pd.offsets.BDay(1)], columns=[f'Predicted Price (D+{i})' for i in range(0, 10)])])

#Final Graph
st.subheader('prediction vs original')
fig=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='original price')
plt.plot(y_predicted,'r',label='predicted price')
blue_line = Line2D([0], [0], color='b', label='Original Price')
red_line = Line2D([0], [0], color='r', label='Predicted Price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend(handles=[blue_line, red_line])
plt.show()

# plot the graph, write the MAPE, and plot the actual vs predicted price table
st.pyplot(fig)
st.write(f'Mean Absolute Percentage Error: {mape:.2f}%')
y_test=df.Close.sort_index(ascending=False)
result_df = pd.concat([y_test, y_predicted], axis=1)
result_df.rename(columns={'Close': 'Original Price'}, inplace=True)
st.write(result_df.sort_index(ascending=False).head(6))