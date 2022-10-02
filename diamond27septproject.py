#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


df=pd.read_csv("diamonds.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.head(10)


# In[6]:


df.tail(10)


# In[7]:


df.shape


# # EDA columns / null / duplicates / outliers / treatement / dummies / X,y / split,train,test'''

# In[8]:


df=df.drop("Unnamed: 0", axis = 1)


# In[9]:


df.shape


# In[10]:


df.columns


# In[11]:


df.isna().sum()


# In[12]:


df.info()


# In[13]:


df.duplicated().sum()


# In[14]:


df.nunique()


# In[15]:


df["cut"].unique()


# In[16]:


df["cut"].value_counts()


# In[17]:


plt.boxplot(df["carat"])


# In[18]:


df.nunique()


# In[19]:


df["cut"].unique() # for checking unique values


# In[20]:


df["cut"].value_counts() # for checking number of unique values


# In[21]:


df["cut"].value_counts().plot(kind="bar")


# In[22]:


df.describe(percentiles = [0.01,0.02,0.03,0.04,0.05,.06,.07,.08,.09,.91,.92,.93,.94,.95,.96,.97,.98,.99]).T


# In[23]:


plt.boxplot(df["carat"])


# In[24]:


df["carat"]=np.where(df["carat"]>2.18,2.18,df["carat"] ) 


# In[25]:


plt.boxplot(df["carat"])


# In[26]:


plt.boxplot(df["depth"])


# In[27]:


df["depth"]=np.where(df["depth"]>65.60,65.60,df["depth"] ) 
df["depth"]=np.where(df["depth"]<57.90,57.90,df["depth"] )


# In[28]:


plt.boxplot(df["depth"])


# In[29]:


plt.boxplot(df["table"])


# In[30]:


df["table"]=np.where(df["table"]>64.00,64.00,df["table"] ) 
df["table"]=np.where(df["table"]<53.00,53.00,df["table"] )


# In[31]:


plt.boxplot(df["table"])


# In[32]:


plt.boxplot(df["price"])


# In[33]:


df["price"]=np.where(df["price"]>17378.22,17378.22,df["price"] ) 


# In[34]:


plt.boxplot(df["price"])


# In[35]:


plt.boxplot(df["x"])


# In[36]:


df["x"]=np.where(df["x"]<4.02,4.02,df["x"] ) 


# In[37]:


plt.boxplot(df["x"])


# In[38]:


plt.boxplot(df["y"])


# In[39]:


df["y"]=np.where(df["y"]>8.34,8.34,df["y"] ) 
df["y"]=np.where(df["y"]<4.04,4.04,df["y"] ) 


# In[40]:


plt.boxplot(df["y"])


# In[41]:


plt.boxplot(df["z"])


# In[42]:


df["z"]=np.where(df["z"]>5.15,5.15,df["z"] ) 
df["z"]=np.where(df["z"]<2.48,2.48,df["z"] ) 


# In[43]:


plt.boxplot(df["z"])


# In[44]:


df.info()


# # create dummies of objects beacuse machine can only understand binary language

# In[45]:


df.info()


# In[46]:


df1 = pd.get_dummies(df, columns = ["cut","color" , "clarity"], drop_first = True)  


# In[47]:


df1


# In[48]:


y=df1["price"]  # dependent variable price
X=df1.drop(columns=["price"])


# In[49]:


X_train , X_test, y_train , y_test = train_test_split (X,y, test_size = 0.2 , random_state=0) 


# In[50]:


lr = LinearRegression()


# In[51]:


lr.fit(X_train,y_train)


# In[52]:


lr.score(X_train,y_train)


# In[53]:


print("Train accuracy" , lr.score(X_train,y_train))
print("Test accuracy" , lr.score(X_test , y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




