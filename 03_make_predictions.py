#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# In[2]:


from tensorflow.keras.preprocessing.image import load_img, img_to_array

# In[3]:


# load model
model = tf.keras.models.load_model('./models/object_detection.h5')
print('model loaded sucessfully')

# In[4]:


path = './test_images/N207.jpeg'
image = load_img(path) # PIL object
image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
image1 = load_img(path,target_size=(224,224))
image_arr_224 = img_to_array(image1)/255.0  # convert into array and get the normalized output

# In[5]:


# size of the orginal image
h,w,d = image.shape
print('Height of the image =',h)
print('Width of the image =',w)

# In[6]:


plt.figure(figsize=(10,8))
plt.imshow(image)
plt.show()

# In[7]:


image_arr_224.shape

# In[8]:


test_arr = image_arr_224.reshape(1,224,224,3)
test_arr.shape

# In[9]:


# make predictions
coords = model.predict(test_arr)
coords

# In[10]:


# denormalize the values
denorm = np.array([w,w,h,h])
coords = coords * denorm
coords

# In[11]:


coords = coords.astype(np.int32)
coords

# In[12]:


# draw bounding on top the image
xmin, xmax,ymin,ymax = coords[0]
pt1 =(xmin,ymin)
pt2 =(xmax,ymax)
print(pt1, pt2)
cv2.rectangle(image,pt1,pt2,(0,255,0),3)

plt.figure(figsize=(10,8))
plt.imshow(image)
plt.show()

# In[13]:


# create pipeline
path = './test_images/N207.jpeg'
def object_detection(path):
    # read image
    image = load_img(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    image1 = load_img(path,target_size=(224,224))
    # data preprocessing
    image_arr_224 = img_to_array(image1)/255.0  # convert into array and get the normalized output
    h,w,d = image.shape
    test_arr = image_arr_224.reshape(1,224,224,3)
    # make predictions
    coords = model.predict(test_arr)
    # denormalize the values
    denorm = np.array([w,w,h,h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    # draw bounding on top the image
    xmin, xmax,ymin,ymax = coords[0]
    pt1 =(xmin,ymin)
    pt2 =(xmax,ymax)
    print(pt1, pt2)
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    return image, coords

# In[14]:


path = './test_images/N147.jpeg'
image, cods = object_detection(path)

plt.figure(figsize=(10,8))
plt.imshow(image)
plt.show()

# # Optical Character Recognition - OCR

# In[15]:


import pytesseract as pt
pt.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# In[21]:


import pytesseract as pt
pt.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

path = './test_images/N207.jpeg'
image, cods = object_detection(path)

img = np.array(load_img(path))
xmin ,xmax,ymin,ymax = cods[0]
roi = img[ymin:ymax,xmin:xmax]

# extract text from image
text = pt.image_to_string(roi)
print(text)


# In[17]:


img = np.array(load_img(path))
xmin ,xmax,ymin,ymax = cods[0]
roi = img[ymin:ymax,xmin:xmax]

# In[18]:


plt.imshow(roi)
plt.show()

# In[19]:


# extract text from image
text = pt.image_to_string(roi)
print(text)

# In[ ]:




# In[ ]:




# In[ ]:



