#!/usr/bin/env python
# coding: utf-8

# <h1>AutoEncoder</h1>

# <h4>Importing the libraries</h4>

# In[1]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import variable


# <h4>Importing the dataset</h4>

# In[2]:


movies = pd.read_csv(
    'ml-1m/movies.dat',
    sep='::',
    engine='python',
    encoding='latin-1',
    names=['MovieID', 'Title', 'Genres']
)


# In[3]:


movies.head()


# In[4]:


users = pd.read_csv(
    'ml-1m/users.dat',
    sep='::',
    engine='python',
    encoding='latin-1',
    names = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
)


# In[5]:


users.head()


# In[6]:


ratings = pd.read_csv(
    'ml-1m/ratings.dat',
    sep='::',
    engine='python',
    encoding='latin-1',
    names =  ['UserID', 'MovieID', 'Rating', 'Timestamp']
)


# In[7]:


ratings.head()


# <h4>Preparing the training set and test set</h4>

# In[20]:


# Define column names as per MovieLens 100K format
column_names = ['user_id', 'movie_id', 'rating', 'timestamp']

# Load the dataset with the correct tab delimiter
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t', names=column_names)


# In[21]:


training_set


# In[22]:


training_set = np.array(training_set, dtype = 'int')


# In[23]:


training_set


# In[24]:


# Define column names as per MovieLens 100K format
column_names = ['user_id', 'movie_id', 'rating', 'timestamp']

# Load the dataset with the correct tab delimiter
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t', names=column_names)


# In[25]:


test_set


# In[26]:


test_set = np.array(training_set, dtype = 'int')


# In[27]:


test_set


# <h4>Getting the number of user and movies</h4>

# In[28]:


nb_users = int(max(training_set[:, 0].max(), test_set[:, 0].max()))


# In[29]:


nb_users


# In[30]:


nb_movies = int(max(training_set[:, 1].max(), test_set[:, 1].max()))


# In[31]:


nb_movies


# <h4>Converting the data into an array with users in lines and movies in columns</h4>

# In[32]:


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

# In[ ]:


training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# <h4>Creating the architecture of the nerual network</h4>

# In[35]:


class SAE(nn.Module):
    def __init__(self,):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies,20)
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)


# <h4>Training the SAE</h4>

# In[37]:


nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = torch.tensor(training_set[id_user], dtype=torch.float).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target = target.detach()
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))


# <h4>Testing the SAE</h4>

# In[42]:


test_loss = 0
s = 0.
for id_user in range(nb_users):
         input = torch.tensor(training_set[id_user], dtype=torch.float).unsqueeze(0)
         target = torch.tensor(test_set[id_user], dtype=torch.float).unsqueeze(0)
         if torch.sum(target.data > 0) > 0:
             output = sae(input)
             target = target.detach()
             output[target == 0] = 0
             loss = criterion(output, target)
             mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
             test_loss += np.sqrt(loss.item() * mean_corrector)
             s += 1.
print('test_loss: '+str(test_loss/s)) 


# In[ ]:




