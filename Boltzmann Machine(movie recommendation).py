#!/usr/bin/env python
# coding: utf-8

# <h1>Boltzmann Machine</h1>

# <h4>Importing the libraries</h4>

# In[3]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import variable


# <h4>Importing the dataset</h4>

# In[4]:


movies = pd.read_csv(
    'ml-1m/movies.dat',
    sep='::',
    engine='python',
    encoding='latin-1',
    names=['MovieID', 'Title', 'Genres']
)


# In[5]:


movies.head()


# In[6]:


users = pd.read_csv(
    'ml-1m/users.dat',
    sep='::',
    engine='python',
    encoding='latin-1',
    names = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
)


# In[7]:


users.head()


# In[8]:


ratings = pd.read_csv(
    'ml-1m/ratings.dat',
    sep='::',
    engine='python',
    encoding='latin-1',
    names =  ['UserID', 'MovieID', 'Rating', 'Timestamp']
)


# In[9]:


ratings.head()


# <h4>Preparing the training set and test set</h4>

# In[10]:


# Define column names as per MovieLens 100K format
column_names = ['user_id', 'movie_id', 'rating', 'timestamp']

# Load the dataset with the correct tab delimiter
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t', names=column_names)


# In[11]:


training_set


# In[12]:


training_set = np.array(training_set, dtype = 'int')


# In[13]:


training_set


# In[14]:


# Define column names as per MovieLens 100K format
column_names = ['user_id', 'movie_id', 'rating', 'timestamp']

# Load the dataset with the correct tab delimiter
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t', names=column_names)


# In[15]:


test_set


# In[16]:


test_set = np.array(training_set, dtype = 'int')


# In[17]:


test_set


# <h4>Getting the number of user and movies</h4>

# In[18]:


nb_users = int(max(training_set[:, 0].max(), test_set[:, 0].max()))


# In[19]:


nb_users


# In[20]:


nb_movies = int(max(training_set[:, 1].max(), test_set[:, 1].max()))


# In[21]:


nb_movies


# <h4>Converting the data into an array with users in lines and movies in columns</h4>

# In[22]:


def convert (data):
    new_data = []
    for id_users in range (1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)


# <h4>Converting the data into torch tensor</h4>

# In[23]:


training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# <h4>Converting the ratings into binary ratings 1 (Liked) or 0 (not liked)</h4>

# In[24]:


training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


# In[25]:


training_set


# <h4>Creating the architecture of the nerual network</h4>

# In[26]:


class RBM ():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1,nh)
        self.b = torch.randn(1,nv)
    def  sample_h (self,x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v (self,y):
        wy = torch.mm(y,self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train (self, v0, vk,ph0, phk):
        self.W += torch.mm(v0.t(), ph0).t() - torch.mm(vk.t(), phk).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk),0)
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv,nh) 


# <h4>Training the RBM</h4>

# In[27]:


nb_epoch = 10
for epoch in range (1, nb_epoch+1):
   train_loss = 0
   s = 0.  
   for id_user in range (0, nb_users - batch_size, batch_size):
       vk = training_set[id_user : id_user + batch_size]
       v0 = training_set[id_user : id_user + batch_size]
       ph0,_ = rbm.sample_h(v0)
       for K in range (10):
           _,hk = rbm.sample_h(vk)
           _,vk = rbm.sample_v(hk)
           vk[v0<0] = v0[v0<0]
       phk,_ = rbm.sample_h(vk)
       rbm.train(v0, vk, ph0, phk)
       train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
       s += 1.
   print('eposide: ' +str(epoch)+ 'loss: ' +str(train_loss/s))
   
# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))


# In[ ]:




