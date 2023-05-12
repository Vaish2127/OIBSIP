#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


df = pd.read_csv("/Users/vaishnavisomankar/Downloads/Iris.csv.xls")
df


# In[24]:


df.describe()


# In[6]:


df["Species"].unique()


# In[7]:


df.groupby("Species").size()


# In[8]:


corr = df.corr(numeric_only = True)
plt.subplots(figsize=(10,6))
sns.heatmap(corr, annot=True)


# In[9]:


corr = df.corr(numeric_only = True)
plt.subplots(figsize=(10,6))
sns.heatmap(corr, annot=True)


# In[10]:


corr = df.corr(numeric_only = True)
plt.subplots(figsize=(10,6))
sns.heatmap(corr, annot=True)


# In[11]:


corr = df.corr()
plt.subplots(figsize=(10,6))
sns.heatmap(corr, annot=True)


# In[13]:


X = df.iloc[:, 1:5].values
y = df.iloc[:, 5].values


# In[14]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
y


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[16]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[17]:


plt.figure()
sns.pairplot(df.drop("Id", axis=1), hue = "Species", height=3, markers=["o", "s", "D"])
plt.sh


# In[18]:


plt.figure()
sns.pairplot(df.drop("Id", axis=1), hue = "Species", height=3, markers=["o", "s", "D"])
plt.show()


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score


# In[20]:


classifier = KNeighborsClassifier(n_neighbors=3)


# In[21]:


classifier.fit(X_train, y_train)


# In[22]:


y_pred = classifier.predict(X_test)


# In[23]:


accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy: ' + str(round(accuracy, 2)) + '%')


# In[ ]:




