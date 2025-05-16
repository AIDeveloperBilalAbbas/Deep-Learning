#!/usr/bin/env python
# coding: utf-8

# <h3>Recurrent Nerual Network</h3>

# <h4>Part 1- Data preprocessing</h4>

# <h5>Importing the libraries</h5>

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# <h5>Importing the training set</h5>

# In[65]:


dataset_train = pd.read_csv('Google_Stock_Price_Train.csv', encoding='utf-8')
training_set = dataset_train.iloc[:, 1:2].values


# In[66]:


print(dataset_train)  # Check the first 5 rows
print(len(dataset_train))  # Verify the number of rows


# In[67]:


training_set


# <h5>Feature Scaling</h5>

# In[68]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
training_set_scaled = sc.fit_transform(training_set)


# In[69]:


training_set_scaled


# <h5>Creating a data structure with 60 timesteps and 1 output</h5>

# In[70]:


X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)        


# In[71]:


X_train


# <h5>Reshaping</h5>

# In[72]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))


# In[73]:


X_train


# <h3>Part 2- Building the RNN</h3>

# <h5>Importing the keras libraries and packages</h5>

# In[74]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Input


# <h5>Initializing the RNN</h5>

# In[75]:


regressor = Sequential()


# <h5> Define Input layer as the first layer</h5>

# In[76]:


regressor.add(Input(shape=(X_train.shape[1], 1)))


# <h5>Adding the first LSTM layer and some Dropout regularization</h5>

# In[77]:


regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))


# <h5>Adding the Second LSTM layer and some Dropout regularization</h5>

# In[78]:


regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))


# <h5>Adding the third LSTM layer and some Dropout regularization</h5>

# In[79]:


regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))


# <h5>Adding the Fourth LSTM layer and some Dropout regularization</h5>

# In[80]:


regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))


# <h5>Adding the output layer</h5>

# In[81]:


regressor.add(Dense(units = 1))


# <h5>Compiling the RNN</h5>

# In[82]:


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# <h5># Fitting the RNN to training set</h5>

# In[83]:


regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# <h3>part 3- Making the prediction and visualizing the result</h3>

# <h5>Getting the real stock price of 2017</h5>

# In[107]:


dataset_test = pd.read_csv('Google_Stock_Price_Test.csv', encoding='utf-8')
real_stock_price = dataset_test.iloc[:, 1:2].values
print(len(real_stock_price))


# <h5>Getting the predicted stock price of 2017</h5>

# In[108]:


dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']), axis = 0)
inputs = dataset_total.iloc[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)  

# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[109]:


# print("Total length:", len(dataset_total))
# print("Test length:", len(dataset_test))
# print("Start index:", len(dataset_total) - len(dataset_test) - 60)
# print("inputs shape:", inputs.shape)
print(len(predicted_stock_price))


# <h4>Visualizing the reuslt</h4>

# In[110]:


plt.plot(predicted_stock_price, color= 'blue', label = 'Predicted Google Stock Price')
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()


# In[111]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))  # Larger figure size

# Plotting with styles and markers
plt.plot(predicted_stock_price, color='blue', linewidth=2, linestyle='--', marker='o', label='Predicted Google Stock Price')
plt.plot(real_stock_price, color='red', linewidth=2, linestyle='-', marker='x', label='Real Google Stock Price')

# Titles and labels
plt.title('Google Stock Price Prediction (Jan 2017)', fontsize=16, fontweight='bold')
plt.xlabel('Time (Days)', fontsize=14)
plt.ylabel('Google Stock Price (USD)', fontsize=14)

# Grid and legend
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', fontsize=12)

# Tight layout and showing the plot
plt.tight_layout()
plt.show()


# In[113]:


import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(rmse)


# Hi guys,
# 
# as seen in the practical lectures, the RNN we built was a regressor. Indeed, we were dealing with Regression because we were trying to predict a continuous outcome (the Google Stock Price). For Regression, the way to evaluate the model performance is with a metric called RMSE (Root Mean Squared Error). It is calculated as the root of the mean of the squared differences between the predictions and the real values.
# 
# However for our specific Stock Price Prediction problem, evaluating the model with the RMSE does not make much sense, since we are more interested in the directions taken by our predictions, rather than the closeness of their values to the real stock price. We want to check if our predictions follow the same directions as the real stock price and we don’t really care whether our predictions are close the real stock price. The predictions could indeed be close but often taking the opposite direction from the real stock price.
# 
# Nevertheless if you are interested in the code that computes the RMSE for our Stock Price Prediction problem, please find it just below:
# 
# import math
# from sklearn.metrics import mean_squared_error
# rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
# Then consider dividing this RMSE by the range of the Google Stock Price values of January 2017 (that is around 800) to get a relative error, as opposed to an absolute error. It is more relevant since for example if you get an RMSE of 50, then this error would be very big if the stock price values ranged around 100, but it would be very small if the stock price values ranged around 10000.
# 
# Enjoy Deep Learning!

# In[ ]:




