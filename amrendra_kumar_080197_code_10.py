#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt
import seaborn as sns
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler


# In[2]:


df=pd.read_csv("data.csv")


# In[3]:


df.shape


# In[4]:


df


# In[5]:


df.columns


# In[6]:


del df["date_of_game"]
del df["game_season"]
del df["Unnamed: 0"]


# In[7]:


df.shape


# In[8]:


df["type_of_combined_shot"].value_counts()


# In[9]:


df["range_of_shot"].value_counts()


# In[10]:


df=df.fillna({"range_of_shot":"Less Than 8 ft."})


# In[15]:


df["range_of_shot"]=df["range_of_shot"].astype("category")


# In[16]:


df["range_of_shot_cat"]=df["range_of_shot"].cat.codes


# In[17]:


del df["range_of_shot"]


# In[18]:


df["area_of_shot"].value_counts()


# In[12]:


df=df.fillna({"area_of_shot":"Center(C)"})


# In[13]:


df["area_of_shot"]=df["area_of_shot"].astype("category")


# In[19]:


df["area_of_shot_cat"]=df["area_of_shot"].cat.codes


# In[20]:


del df["area_of_shot"]


# In[21]:


df["shot_basics"].value_counts()


# In[22]:


df=df.fillna({"shot_basics":"Mid Range"})


# In[23]:


df["shot_basics"]=df["shot_basics"].astype("category")


# In[24]:


df["shot_basics_cat"]=df["shot_basics"].cat.codes


# In[25]:


del df["shot_basics"]


# In[26]:


df.shape


# In[27]:


train=df[pd.notnull(df["is_goal"])]


# In[28]:


x_train=train.loc[:,df.columns != "is_goal"]


# In[29]:


y_train=train["is_goal"]


# In[30]:


k=x_train.select_dtypes([np.number])


# In[71]:


k=k.fillna(method="ffill")


# In[72]:


k.isnull().sum().sum()


# In[73]:


m=y_train


# In[74]:


from sklearn.cross_validation import train_test_split


# In[75]:


X_Train,X_Test,Y_Train,Y_Test=train_test_split(k,m,random_state=0)


# In[76]:


sc = StandardScaler()


# In[77]:


k_train_std = sc.fit_transform(X_Train)


# In[78]:


m_test_std = sc.transform(X_Test)


# In[103]:


from sklearn.linear_model import LogisticRegression


# In[104]:


lg=LogisticRegression(C=100)


# In[105]:


lg.fit(X_Train,Y_Train)


# In[106]:


lg.score(X_Train,Y_Train)


# In[107]:


test=df[pd.isnull(df["is_goal"])]


# In[108]:


x_test=test.loc[:,df.columns != "is_goal"]


# In[109]:


index_new=x_test.index


# In[110]:


index_new.shape


# In[111]:


kt=x_test.select_dtypes([np.number])


# In[112]:


kt=kt.fillna(method="ffill")


# In[113]:


probability = lg.predict_proba(kt)[:,1]


# In[114]:


probability


# In[115]:


b=pd.Series(probability)


# In[116]:


b.index=index_new+1


# In[117]:


b


# In[119]:


b.to_csv("final_valueee_10.csv")


# In[120]:


print(b)


# In[ ]:




