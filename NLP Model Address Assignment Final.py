#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd


# # Opening the Data

# In[2]:


sample_address = gpd.GeoDataFrame.from_file('Sample Address Fix.geojson')


# In[3]:


sample_address.head()


# In[4]:


sample_address['Label'] = sample_address['Zona'].apply(lambda x: 'ZP '+str(x))


# In[5]:


sample_address.head(10)


# In[6]:


sample_address = sample_address[sample_address['Zona'] < 6]


# # Text Preprocessing

# The problem with text data is, computers can't directly process it. Therefore, we need to transform the data into something that computers are capable of processing. The simplest method is using the Bag of Words approach. Bag of Words approach, it's done by converting text into a list containing the number of appearance of each words.

# ### 1. Tokenizing

# First we need to 'tokenize' the text. Tokenizing is a process of converting a document or a text into a list of words in the text. Let's create a function to do that.

# In[10]:


import string


# In[9]:


def text_process(text):
    cleantxt = [char for char in text if char not in string.punctuation]
    cleantxt =  "".join(cleantxt)
    return [word for word in cleantxt.split()]


# This function split each word, and remove all the punctuations.

# In[11]:


text_process('Nama saya budi, saya suka bermain bola!')


# Now, let's try applying the tokenizer function into our data

# ##### Before:

# In[12]:


sample_address['address'].head(5)


# ##### After:

# In[13]:


sample_address['address'].head(5).apply(text_process)


# ### 2. Vectorization

# After tokenizing the text, we will put the text into a vector containing each word count. All the text then will be stored in a data called _Sparse Matrix_.
# 
# For example:
# 
# <table border = “1“>
# <tr>
# <th></th> <th>Address 1</th> <th>Address 2</th> <th>...</th> <th>Address N</th> 
# </tr>
# <tr>
# <td><b>Word 1 Count</b></td><td>0</td><td>1</td><td>...</td><td>0</td>
# </tr>
# <tr>
# <td><b>Word 2 Count</b></td><td>0</td><td>0</td><td>...</td><td>0</td>
# </tr>
# <tr>
# <td><b>...</b></td> <td>1</td><td>2</td><td>...</td><td>0</td>
# </tr>
# <tr>
# <td><b>Word N Count</b></td> <td>0</td><td>1</td><td>...</td><td>1</td>
# </tr>
# </table>
# 
# 
# 
# To do that, we will need to use a module from Scikit-Learn called CountVectorizer

# ##### Import CountVectorizer and Create a CountVectorizer Object
# 
# Use the _text_process_ function that we created earlier as analyzer

# In[14]:


from sklearn.feature_extraction.text import CountVectorizer


# In[15]:


bow_transformer = CountVectorizer(analyzer=text_process).fit(sample_address['address'])


# In[16]:


bow = bow_transformer.transform(sample_address['address'])


# In[17]:


print(bow.shape)


# # Training the Model
# 
# We will use Multinomial Naive Bayes classifier from Scikit Learn module to create a machine learning object and train the model

# In[18]:


from sklearn.naive_bayes import MultinomialNB


# In[19]:


zp_model = MultinomialNB()


# In[20]:


zp_model.fit(bow,sample_address['Label'])


# # Creating Data Pipeline

# To get the Model reusable, and we don't need to reprocess each text we use in the model, we will create a data Pipeline consisting of the Steps we've done.

# In[21]:


from sklearn.pipeline import Pipeline


# In[22]:


model_pipeline = Pipeline([('bow',CountVectorizer(analyzer=text_process)),
                          ('classifier', MultinomialNB())])


# # Train Test Split

# Let's split our data into train data and test data, consisting of 80% training data and 20% test Data

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


sample_address.head()


# In[25]:


X = sample_address['address']
y = sample_address['Label']


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# # Fitting the Data into the Pipeline

# In[27]:


model_pipeline.fit(X_train,y_train)


# In[28]:


pred = model_pipeline.predict(X_test)


# # Model Evaluation

# In[29]:


from sklearn.metrics import classification_report


# In[30]:


print(classification_report(y_test,pred))


# # Adjusting the Model

# ### Using TF-IDF

# In[31]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[32]:


adj_model1 = Pipeline([('bow',CountVectorizer(analyzer=text_process)),
                       ('tfidf',TfidfTransformer()),
                       ('classifier', MultinomialNB())])


# In[33]:


adj_model1.fit(X_train,y_train)


# In[34]:


pred_adj1 = adj_model1.predict(X_test)


# In[35]:


print(classification_report(y_test,pred_adj1))


# ### Adjusting Text Processor

# In[36]:


def text_process1(text):
    cleantxt = []
    for char in text:
        if char in ['/','-']:
            cleantxt.append(' ')
            continue
        elif char in string.punctuation:
            continue
        else: cleantxt.append(char)
    cleantxt =  "".join(cleantxt)
    cleantxt = [word for word in cleantxt.split(',') if word.lower() != 'jakarta utara']
    cleantxt =  "".join(cleantxt)
    return [word for word in cleantxt.split() if word.lower() not in ('jalan','jl')]


# In[37]:


adj_model2 = Pipeline([('bow',CountVectorizer(analyzer=text_process1)),
                       ('tfidf',TfidfTransformer()),
                       ('classifier', MultinomialNB())])


# In[38]:


adj_model2.fit(X_train, y_train)


# In[39]:


pred_adj2 = adj_model2.predict(X_test)


# In[40]:


print(classification_report(y_test,pred_adj2))


# # Choosing Different Classifier

# #### Using Random Forest

# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[42]:


rf_model = Pipeline([('bow',CountVectorizer(analyzer=text_process1)),
                       ('tfidf',TfidfTransformer()),
                       ('classifier', RandomForestClassifier())])


# In[43]:


rf_model.fit(X_train,y_train)


# In[44]:


rf_pred = rf_model.predict(X_test)


# In[45]:


print(classification_report(y_test,rf_pred))


# Random forest has significantly lower accuracy

# #### Using _k_-Nearest Neighbor

# In[46]:


from sklearn.neighbors import KNeighborsClassifier


# In[47]:


knn_model = Pipeline([('bow',CountVectorizer(analyzer=text_process1)),
                       ('tfidf',TfidfTransformer()),
                       ('classifier', KNeighborsClassifier(n_neighbors=1))])


# In[48]:


knn_model.fit(X_train,y_train)


# In[49]:


knn_pred = knn_model.predict(X_test)


# In[50]:


print(classification_report(y_test,knn_pred))


# #### Choosing _k_ Value

# In[51]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


error_rate = []

for i in range(1,40):
    
    knn = Pipeline([('bow',CountVectorizer(analyzer=text_process1)),
                       ('tfidf',TfidfTransformer()),
                       ('classifier', KNeighborsClassifier(n_neighbors=i))])
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[68]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.savefig('K-Value.jpg', dpi=None, facecolor='w', edgecolor='w',
          orientation='landscape', papertype=None, format=None,
          transparent=False, bbox_inches=None, pad_inches=0.1,
          frameon=None, metadata=None)


# We can see that the error amount considered low around k amount of 10, we will take the value of k as 10

# In[73]:


knn_model = Pipeline([('bow',CountVectorizer(analyzer=text_process1)),
                       ('tfidf',TfidfTransformer()),
                       ('classifier', KNeighborsClassifier(n_neighbors=10))])


# In[74]:


knn_model.fit(X_train,y_train)


# In[75]:


knn_pred = knn_model.predict(X_test)


# In[76]:


print(classification_report(y_test,knn_pred))


# In[ ]:




