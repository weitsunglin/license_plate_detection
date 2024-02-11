#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


import xml.etree.ElementTree as xet

# In[5]:


from glob import glob

# In[6]:


path = glob('./images/*.xml')

# In[7]:


path

# In[20]:


# filename = path[0]

labels_dict = dict( filepath = [], xmin = [],xmax =[],ymin = [],ymax = [] )
for filename in path:
    info = xet.parse(filename)
    root = info.getroot()
    member_object = root.find( 'object' )
    lables_info = member_object.find( 'bndbox' )
    xmin = int(lables_info.find( 'xmin' ).text)
    xmax = int(lables_info.find( 'xmax' ).text)
    ymin = int(lables_info.find( 'ymin' ).text)
    ymax = int(lables_info.find( 'ymax' ).text)
    print( xmin, xmax , ymin , ymax)
    labels_dict['filepath'].append(filename)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)


# In[22]:


df = pd.DataFrame(labels_dict)
df


# In[23]:


df.to_csv('labels.csv',index=False)

# In[ ]:



