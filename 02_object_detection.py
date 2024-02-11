#!/usr/bin/env python
# coding: utf-8

# In[72]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

# In[73]:


df = pd.read_csv('labels.csv')
df.head()

# In[74]:


import xml.etree.ElementTree as xet

# In[75]:


filename = df['filepath'][0]
filename

# In[76]:


def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('./images',filename_image)
    return filepath_image

# In[77]:


getFilename(filename)

# In[78]:


image_path = list(df['filepath'].apply(getFilename))
image_path 

# #### verify image and output

# In[79]:


file_path = image_path[0]
file_path 

# In[80]:


img = cv2.imread(file_path)

cv2.namedWindow('example',cv2.WINDOW_NORMAL)
cv2.imshow('example',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In[81]:


# 1093	1396	645	727
cv2.rectangle(img,(1093,645),(1396,727),(0,255,0),3)
cv2.namedWindow('example',cv2.WINDOW_NORMAL)
cv2.imshow('example',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ### Data Preprocessing

# In[82]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# In[83]:


labels = df.iloc[:,1:].values

# In[84]:


data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    h,w,d = img_arr.shape
    # prepprocesing
    load_image = load_img(image,target_size=(224,224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr/255.0 # normalization
    # normalization to labels
    xmin,xmax,ymin,ymax = labels[ind]
    nxmin,nxmax = xmin/w,xmax/w
    nymin,nymax = ymin/h,ymax/h
    label_norm = (nxmin,nxmax,nymin,nymax) # normalized output
    # -------------- append
    data.append(norm_load_image_arr)
    output.append(label_norm)

# In[85]:


X = np.array(data,dtype=np.float32)
y = np.array(output,dtype=np.float32)

# In[86]:


X.shape,y.shape

# In[87]:


x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

# ### Deep Learning Model

# In[88]:


from tensorflow.keras.applications import MobileNetV2, InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
import tensorflow as tf

# In[89]:


inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False,
                                     input_tensor=Input(shape=(224,224,3)))
inception_resnet.trainable=False
# ---------------------
headmodel = inception_resnet.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500,activation="relu")(headmodel)
headmodel = Dense(250,activation="relu")(headmodel)
headmodel = Dense(4,activation='sigmoid')(headmodel)

# ---------- model
model = Model(inputs=inception_resnet.input,outputs=headmodel)

# In[90]:


# complie model
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
model.summary()

# ### model training

# In[91]:


from tensorflow.keras.callbacks import TensorBoard

# In[92]:


tfb = TensorBoard('object_detection')

# In[93]:


history = model.fit(x=x_train,y=y_train,batch_size=10,epochs=400,
                    validation_data=(x_test,y_test),callbacks=[tfb],initial_epoch=1)

# In[94]:


model.save('./model/object_detection.h5')

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



