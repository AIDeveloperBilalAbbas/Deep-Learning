#!/usr/bin/env python
# coding: utf-8

# # Convolutional Nerual Network

# Importing the libraries

# In[1]:


import tensorflow as tf
print(tf.__version__)


# In[2]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[3]:


tf.__version__


# # Part 1 - Data preprocessing

# Preprocessing the training set

# In[9]:


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',  # this is the target directory
        target_size=(64, 64),  # all images will be resized to 150x150
        batch_size= 32,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels


# Preprocessing the Test set

# In[12]:


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)
# this is a similar generator, for validation data
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size= 32,
        class_mode='binary')


# ## Part 2 - Building the CNN

# Initializing the CNN

# In[16]:


cnn = tf.keras.models.Sequential()


# Step 1- Convolution

# In[19]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,3]))


# step 2- Pooling

# In[22]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# Adding a second convolutional Layer

# In[25]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# Step 3- Flattening

# In[28]:


cnn.add(tf.keras.layers.Flatten())


# Step 4- Full Connection

# In[31]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# Step 5- Output Layer

# In[34]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# # Part 3 - Training the CNN

# Compiling the CNN

# In[38]:


cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Training the CNN on the training set and evaluating it on the test set

# In[43]:


cnn.fit(x= training_set, validation_data=test_set, epochs=25)


# # Part 4- Making a single prediction

# In[59]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_5.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'

print(prediction)


# In[ ]:




