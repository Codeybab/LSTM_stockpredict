#!/usr/bin/env python
# coding: utf-8

# In[67]:


import matplotlib.pyplot as plt
import math
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import keras 
from keras.models import Sequential 
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')
df = pd.read_csv('XAU5Y.csv')
df


# In[68]:


#seeing the data 
plt.figure(figsize=(16,6))
plt.title('close price')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close price USD', fontsize=18)
plt.show()


# In[70]:


#create data frame with close 
data = df.filter(['Close'])
dataset = data.values
#getting rows 
training_data_length = math.ceil(len(dataset)*.8)
training_data_length
#scaling 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
training_data_length, scaled_data


# In[71]:


#create training dataset for scaled data 
train_data = scaled_data[0:training_data_length, :]
x_train = []
y_train = []
for i in range(120, len(train_data)):
    x_train.append(train_data[i-120:i,0])
    y_train.append(train_data[i,0])
    if i<= 121:
        print(x_train)
        print(y_train)
        print()
#making trainable arrays 
x_train, y_train = np.array(x_train), np.array(y_train)
x_train= np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_train.shape


# In[72]:


#build model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
#train model 
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[75]:


#testing dataset 
#897 to 1271
test_data = scaled_data[training_data_length - 120: , :]
x_test = []
y_test= dataset[training_data_length:, :]
for i in range(120, len(test_data)):
    x_test.append(test_data[i-120:i,0])
#back to numpy array 
x_test = np.array(x_test)
#make data 3d 
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))


# In[79]:


#prediction
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
#plots 
train = data[:training_data_length]
validation = data[training_data_length:]
validation['Predictions'] = predictions 
plt.figure(figsize=(14,6))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(train['Close'])
plt.plot(validation[['Close','Predictions']])
plt.legend(['Train', 'Validations', 'Predictions'], loc='lower right')
plt.show(), validation


# In[82]:


#attempting to predict the closing price for 5/31/2021
new_df = df.filter(['Close'])
last_120_days = new_df[-120:].values 
last_120_days_scaled = scaler.transform(last_120_days)
X_test =[]
X_test.append(last_120_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[ ]:





# In[ ]:




