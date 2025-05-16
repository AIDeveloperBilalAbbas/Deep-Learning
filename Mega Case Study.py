#!/usr/bin/env python
# coding: utf-8

# <h2>Merge Case Study - Make a hybrid Deep learning Model</h2>

# <h3>Part 1- Identify the Frauds with self organizing Map</h3>

# <h4>Importing the libraries"</h4>

# In[39]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


# <h4>Importing the dataset</h4>

# In[40]:


dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values


# In[41]:


dataset.head()


# <h4>Feature scaling</h4>

# In[42]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X_Scaled = sc.fit_transform(X)


# <h2>Training the SOM</h2>

# In[43]:


from  minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X_Scaled)
som.train_random(data = X_Scaled, num_iteration = 100)


# <h2>Visualizing the Result</h2>

# In[44]:


from pylab import bone, pcolor, colorbar, plot, show

markers = ['o', 's']  # 0 = legit, 1 = fraud
colors = ['g', 'r']   # 0 = green, 1 = red

plt.figure(figsize=(10, 7))
bone()
pcolor(som.distance_map().T)
colorbar()

plotted = set()
for i, x in enumerate(X_Scaled):
    label = int(y[i])
    w = som.winner(x)
    
    if w not in plotted:
        plot(w[0] + 0.5,
             w[1] + 0.5,
             markers[label],
             markeredgecolor=colors[label],
             markerfacecolor='none',
             markersize=10,
             markeredgewidth=2)
        plotted.add(w)

show()


# <h2>Finding the frauds</h2>

# In[61]:


mappings = som.win_map(X_Scaled)
frauds = np.concatenate((mappings[(5,4)], mappings[(5,3)]), axis=0)
frauds = sc.inverse_transform(frauds)


# In[62]:


fraud_df = pd.DataFrame(frauds)


# In[63]:


fraud_df


# <h2>Part 2- Going from unsupervised to supervised Deep learning</h2>

# <h4>creating the matrix of features</h4>

# In[64]:


customers_raw = dataset.iloc[:, 1:].values


# In[65]:


customers_raw


# <h4>Creating the dependent variable</h4>

# In[66]:


is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1


# In[67]:


# is_fraud


# <h4>Feature Scaling</h4>

# In[68]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers_scaled = sc.fit_transform(customers_raw)


# In[69]:


customers_scaled


# <h2>Building the ANN</h2>

# <h4>Initializing the ANN</h4>

# In[70]:


ann = tf.keras.models.Sequential()


# <h4>Adding the input layer and first hidden layer</h4>

# In[71]:


ann.add(tf.keras.layers.Dense(units=2, kernel_initializer = 'uniform', activation='relu', input_dim = 15))


# <h4>Adding the output layer</h4>

# In[72]:


ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# <h3>Compiling the ANN</h3>

# In[73]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# <h4>Fitting the ANN to training set</h4>

# In[74]:


ann.fit(customers, is_fraud, batch_size = 1, epochs = 2)


# <h2>Making prediction and Evaluating the Model</h2>

# <h4>Predicting the probabilities of frauds </h4>

# In[79]:


y_pred = ann.predict(customers)


# In[80]:


y_pred


# In[81]:


y_pred = np.concatenate((dataset.iloc[:, 0: 1].values, y_pred), axis =1)


# In[82]:


y_pred


# In[83]:


y_pred = y_pred[y_pred[:, 1].argsort()]


# In[84]:


y_pred


# In[ ]:




