#!/usr/bin/env python
# coding: utf-8

# # Data Cleansing

# In[1]:


import pandas as pd
import numpy as np


# First, we will open both scrapped addresses, and make 2 Data Frames objects

# In[2]:


al1 = pd.read_csv('Scraped Address.csv',index_col=0)
al2 = pd.read_csv('Scraped Address1.csv',index_col=0)


# In[3]:


al1.head()


# In[4]:


al2.head()


# To start our data cleansing, let's take a quick look at their features using df.describe() method

# In[5]:


al1.describe()


# In[6]:


al2.describe()


# Then, we will concatenate both Data Frames, and create a new Data Frames that contains all the attributes from both

# In[7]:


fix_add = pd.concat([al1,al2])


# In[8]:


fix_add.head()


# In[9]:


fix_add.describe()


# We can see that from both data we collected, there are about 5000 unique entries. Let's begin to cleanse our data

# # Deleting Unnecessary Features
# 
# As we can see, all the data contains '---' value which are errors on our scraping process. And all the addresses contains word 'Alamat,' in every start of the address.

# In[10]:


fix_add[fix_add['address'] == '---']


# In[11]:


fix_add = fix_add[fix_add['address'] != '---']


# In[12]:


fix_add['address'] = fix_add['address'].apply(lambda x: x.split('Alamat,')[1])


# In[13]:


fix_add.head()


# In[14]:


fix_add.describe()


# In[17]:


fix_add.drop_duplicates(subset='address').describe()


# In[16]:


fix_add.drop_duplicates(subset='address',keep=False).describe()


# In[18]:


fix_al = fix_add.drop_duplicates(subset='address')


# In[28]:


fix_al.head()


# # Joining the DataFrame into ZP Polygon

# In[59]:


no_string(fix_al['coordinate'].iloc[0])


# In[30]:


import geopandas as gpd
import string


# In[58]:


def no_string(text):
    word = [char for char in text if char not in ['(',')',',']]
    word = ''.join(word)
    word = word.split()
    return tuple([float(word[0]), float(word[1])])


# In[61]:


fix_al['coordinate'] = fix_al['coordinate'].apply(no_string)


# In[72]:


fix_al.info()


# In[78]:


fix_al['x'] = [x[0] for x in fix_al['coordinate']]
fix_al['y'] = [x[1] for x in fix_al['coordinate']]


# In[115]:


gdf_al = gpd.GeoDataFrame(fix_al,geometry=gpd.points_from_xy(fix_al.y,fix_al.x))


# In[86]:


gdf_al.to_file('Sample Address.geojson',driver='GeoJSON')


# In[116]:


koja = gpd.GeoDataFrame.from_file("Zona Pengawasan Koja.geojson")


# In[119]:


gdf_fix = gdf_al.sjoin(koja)


# In[117]:


gdf_al = gdf_al.set_crs(epsg=4326)


# In[132]:


gdf_fix = gdf_fix[['address','geometry','Zona']]


# In[133]:


gdf_fix.to_file('Sample Address Fix.geojson',driver='GeoJSON')


# In[ ]:




