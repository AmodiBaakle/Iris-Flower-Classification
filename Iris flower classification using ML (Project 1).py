#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


df=pd.read_csv('Iris.csv') #dataframe
df.head()


# In[8]:


df = df.drop(columns = ['Id'])
df.head()                


# In[9]:


#display stats of data 
df.describe()


# In[10]:


df.info() #display info datatype 


# In[12]:


# display number of sample on each class
df['Species'].value_counts()


# In[14]:


#check null values 
df.isnull().sum()


# In[16]:


df['SepalLengthCm'].hist()


# In[17]:


df['SepalWidthCm'].hist()


# In[18]:


df['PetalLengthCm'].hist()


# In[21]:


df['PetalWidthCm'].hist()


# In[26]:


#scatter plot
colors= ['red', 'yellow', 'blue']
species=['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

for i in  range(3) :
    x=df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal width')
plt.legend()


# In[27]:


#scatter plot
colors= ['red', 'yellow', 'blue']
species=['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

for i in  range(3) :
    x=df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel('Petal Length')
plt.ylabel('Petal width')
plt.legend()


# In[28]:


#scatter plot
colors= ['red', 'yellow', 'blue']
species=['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

for i in  range(3) :
    x=df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend()


# In[30]:


#scatter plot
colors= ['red', 'yellow', 'blue']
species=['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

for i in  range(3) :
    x=df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')
plt.legend()


# In[31]:


#Corealtion matrix
df.corr()


# In[35]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax ,cmap = 'coolwarm')


# In[57]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[58]:


df['Species']=le.fit_transform(df['Species'])
df.head()


# In[59]:


from sklearn.model_selection import train_test_split
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.30)


# In[60]:


#logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[61]:


model.fit(x_train, y_train)


# In[62]:


print("Accuracy : ",model.score(x_test, y_test)*100)


# In[63]:


#knn k nearest neighbors 
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[64]:


model.fit(x_train,y_train)


# In[65]:


print("Accuracy : ",model.score(x_test, y_test)*100)


# In[66]:


#decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[67]:


model.fit(x_train, y_train)


# In[68]:


print("Accuracy : ",model.score(x_test, y_test)*100)


# In[ ]:




