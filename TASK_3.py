#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("/Users/vaishnavisomankar/Downloads/Advertising.csv")
df


# In[3]:


df=df.drop(columns=["Unnamed: 0"])


# In[4]:


df.head()


# In[6]:


df.shape


# In[7]:


x=df.iloc[:,0:-1]
x


# In[8]:


y=df.iloc[:,-1]
y


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)
x_train


# In[10]:


x_test


# In[11]:


y_train


# In[12]:


y_test


# In[13]:


x_train=x_train.astype(int)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
y_test=y_test.astype(int)


# In[15]:


from sklearn.preprocessing import StandardScaler
Sc=StandardScaler()
x_train_scaled=Sc.fit_transform(x_train)


# In[16]:


x_test_scaled=Sc.fit_transform(x_test)


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


lr=LinearRegression()


# In[19]:


lr.fit(x_train_scaled,y_train)


# In[20]:


y_pred=lr.predict(x_test_scaled)


# In[21]:


from sklearn.metrics import r2_score


# In[22]:


r2_score(y_test,y_pred)


# In[23]:


import matplotlib.pyplot as plt


# In[24]:


plt.scatter(y_test,y_pred,c='g')


# In[ ]:




