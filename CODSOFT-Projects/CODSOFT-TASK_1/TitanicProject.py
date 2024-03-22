#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


train_data = pd.read_csv('train.csv')


# In[7]:


train_data.head()


# In[8]:


sns.countplot(x='Survived',data=train_data,hue='Pclass')


# In[9]:


sns.histplot(train_data['Age'],kde=False)


# In[10]:


train_data.info()


# In[11]:


train_data.isnull().sum()


# In[12]:


sns.boxplot(x='Pclass',y='Age',data=train_data)


# # AVERAGES FOR EACH CLASS

# In[13]:


print(train_data[train_data['Pclass'] == 1 ]['Age'].mean())
print(train_data[train_data['Pclass'] == 2 ]['Age'].mean())
print(train_data[train_data['Pclass'] == 3 ]['Age'].mean())


# # REPLACE THE NULL VALUES WITH THE AVERAGES

# In[14]:


def fill_in_na_values(cols):
  age = cols[0]
  pclass = cols[1]
    
  if pd.isnull(age):
    if   pclass == 1:
        return round(train_data[train_data['Pclass'] == 1]['Age'].mean())
    elif pclass == 2:
        return round(train_data[train_data['Pclass'] == 2]['Age'].mean())
    elif pclass == 3:
        return round(train_data[train_data['Pclass'] == 3]['Age'].mean())
  else:
    return age

train_data['Age'] = train_data[['Age', 'Pclass']].apply(fill_in_na_values,axis=1)


# In[15]:


train_data.isnull().sum()


# # VISUALIZE Cabin 

# In[16]:


sns.heatmap(train_data.isnull())


# In[17]:


train_data.head()


# In[19]:


train_data.drop(['Cabin'],axis=1,inplace=True)


# In[20]:


train_data.head()


# In[21]:


train_data.dropna(inplace=True)


# In[22]:


train_data.isnull().sum()


# # Which features must be kept and the ones that be dropped

# In[23]:


train_data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[24]:


train_data.head()


# # One hot-Encode the categorical columns 

# In[35]:


train_data['Embarked'].unique()


# In[48]:


sex = pd.get_dummies(train_data['Sex'],drop_first=True)


# In[49]:


sex


# In[46]:


embarked = pd.get_dummies(train_data['Embarked'],drop_first=True)


# In[47]:


embarked


# In[52]:


train_data.drop(['Sex','Embarked'],axis=1,inplace=True)


# In[53]:


train_data = pd.concat([train_data,sex,embarked],axis=1)


# In[54]:


train_data.head()


# # Scale or Normalize the Data

# In[60]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X = train_data.drop('Survived',axis=1)
y = train_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# In[61]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # We use various models from sklearn (SVM)

# In[63]:


from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)

predictions = svm.predict(X_test)


# # For accuracy

# In[64]:


from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[65]:


from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.5,1,10,50,100,1000],'gamma':[1,0.1,0.001,0.0001,0.00001]}

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)

grid.fit(X_train,y_train)
grid_predictions = grid.predict(X_test)


# In[66]:


print(classification_report(y_test,grid_predictions))
print(confusion_matrix(y_test,grid_predictions))


# # Using another model (Logistic Regression)

# In[67]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

lr_predictions = lr.predict(X_test)

print(classification_report(y_test,lr_predictions))
print(confusion_matrix(y_test,lr_predictions))


# # Using K Nearest Neighbors

# In[77]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

print(classification_report(y_test,knn_predictions))
print(confusion_matrix(y_test,knn_predictions))


# # Using Decision Trees and Random Forest Classifier

# In[81]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dt = DecisionTreeClassifier()
rfc = RandomForestClassifier()

dt.fit(X_train,y_train)

rfc.fit(X_train,y_train)

dt_predictions = dt.predict(X_test)
rfc_predictions = rfc.predict(X_test)


# In[82]:


print(classification_report(y_test,dt_predictions))
print(confusion_matrix(y_test,dt_predictions))


# In[83]:


print(classification_report(y_test,rfc_predictions))
print(confusion_matrix(y_test,rfc_predictions))


# In[ ]:




