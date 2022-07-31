#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
import folium
get_ipython().run_line_magic('matplotlib', 'inline')


# # GeoJson Files as GeoDataFrames in Geopandas

# In[2]:


koja = gpd.GeoDataFrame.from_file('Zona Pengawasan Koja.geojson')


# In[3]:


koja.head()


# # Create Folium Map Object and Adding Overlay

# In[4]:


zp = folium.Map(location=[-6.1163195,106.9136463],zoom_start=13)


# In[5]:


overlay = folium.Choropleth(koja,fill_color='green', key_on= 'zona')


# In[6]:


overlay.add_to(zp)


# In[7]:


zp


# # Calculating Area of Each Supervising Zone
# 
# Convert the geometry into crs and calculate the area in KM^2

# In[8]:


koja['geo_crs'] = koja['geometry'].to_crs({'proj':'cea'})


# In[9]:


koja['area'] = koja['geo_crs'].area / 10 ** 6


# In[10]:


koja['area'].sum()


# In[11]:


koja.head()


# # Get Random Coordinates Based on Area
# 
# I want to get at least 5000 random points. With 53km area, the I would take 100 points every 1KM.

# In[12]:


import numpy as np
import random
from shapely.geometry import Polygon, Point


# In[13]:


def random_coord(poly,num_coord):
    min_x, min_y, max_x, max_y = poly.bounds
    points = []
    
    while len(points) < num_coord:
        random_points = Point([random.uniform(min_x,max_x),random.uniform(min_y,max_y)])
        if random_points.within(poly):
            points.append(random_points)
    return points


# In[22]:


koja[['geometry','area']]


# In[19]:


for poly,num in koja['geometry']koja['area']:
    print(poly,num)


# In[15]:


coords = []

for poly in koja['geometry']:
    rand = random_coord(poly, 100)
    for point in rand:
        coords.append(point)


# In[16]:


xs = [o.y for o in coords]
ys = [o.x for o in coords]


# In[17]:


base_url = "https://www.google.com/maps/search/{0}+,{1}/"


# # Run Selenium

# In[18]:


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from parsel import Selector
from selenium.webdriver.support.ui import WebDriverWait


# In[19]:


driver = webdriver.Chrome(ChromeDriverManager().install(),options=Options().add_argument('--headless'))


# In[20]:


alamat = []
for coord in zip(xs,ys):
    url = base_url.format(coord[0],coord[1])
    driver.get(url)
    driver.implicitly_wait(10)
    try:
        response = Selector(driver.page_source)
        elem = response.xpath('//div[@class="id-content-container"]/div/div')
        al = elem.xpath('//div[contains(@aria-label, "Alamat")]').extract_first('')
        alamat.append({
            'address' : (al.split('\"')[1]),
            'coordinate' : coord})
    except:
        alamat.append({
            'address' : '---',
            'coordinate' : coord})

df = pd.DataFrame(alamat)
df.to_csv('Scraped Address1.csv')


# In[21]:


df.describe()


# In[50]:


df.drop('point',inplace=True,axis=1)


# In[55]:


gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.x,df.y))


# In[53]:


df['x'] = [x[0] for x in df['coordinate']]
df['y'] = [x[1] for x in df['coordinate']]


# In[59]:


gdf = gdf[['address','geometry']]


# In[66]:


koja_new = koja.drop('geo_crs',axis=1)


# In[73]:


def plotDot(point):
    folium.CircleMarker(location=[point],
                       radius=2,
                       weight=5).add_to(zp)


# In[85]:


for point in df['coordinate']:
    folium.CircleMarker(location=[point[0],point[1]],
                       radius=1,
                       weight=5,
                       color='red').add_to(zp)


# In[86]:


zp


# In[22]:


driver.quit()


# In[ ]:




