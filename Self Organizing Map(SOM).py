#!/usr/bin/env python
# coding: utf-8

# <h2>Self Organizing Map</h2>

# <h4>Importing the libraries</h4>

# In[84]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# <h4>Importing the dataset</h4>

# In[85]:


dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values


# In[86]:


dataset.head()


# <h4>Feature Scaling</h4>

# In[87]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X_Scaled = sc.fit_transform(X)


# <h2>Training the SOM</h2>

# In[88]:


from  minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X_Scaled)
som.train_random(data = X_Scaled, num_iteration = 100)


# <h2>Visualizing the result </h2>

# In[89]:


from pylab import bone, pcolor, colorbar, plot, show
# plt.clf()  # Clear previous figure
plt.figure(figsize=(10, 7)) 
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']  # 0 = circle (legit), 1 = square (fraud)
colors = ['g', 'r']   # 0 = green (legit), 1 = red (fraud)
for i, x in enumerate(X_Scaled):
     w = som.winner(x)
     plot(w[0] + 0.5,
       w[1] + 0.5,
       markers[y[i]],
       markeredgecolor=colors[y[i]],
       markerfacecolor='none',   # lowercase 'none' is safer and avoids override
       markersize=10,
       markeredgewidth=2)
show()


# In[90]:


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


# <h2> SOM Interpretation Cheat Sheet</h2>

# SOM Cell Appearance	Marker	Meaning
# 
# 🔲 Light-colored cell	(Any)	⚠️ Anomalous area — neurons here are very different from neighbors → possible fraud zone
# 
# ⚪ Green Circle	Legit (y=0)	✅ Legitimate customer
# 
# 🟥 Red Square	Fraud (y=1)	❌ Confirmed fraudulent customer
# 
# ⚪ Green Circle in Light Cell	Legit (y=0) in Anomaly Zone	⚠️ Legit label but suspicious behavior → potential fraud
# 
# 🟥 Red Square in Light Cell	Fraud (y=1) in Anomaly Zone	❗ Matches fraud label and behavior → confirmed fraud
# 
# (Dark, empty cell)	—	No customer mapped here — normal or unoccupied region

# <h2>Finding the Fraud</h2>

# In[91]:


mappings = som.win_map(X_Scaled)
frauds = np.concatenate((mappings[(4,8)], mappings[(4,7)]), axis=0)
frauds = sc.inverse_transform(frauds)


# In[92]:


fraud_df = pd.DataFrame(frauds)


# In[93]:


fraud_df


# In[ ]:




