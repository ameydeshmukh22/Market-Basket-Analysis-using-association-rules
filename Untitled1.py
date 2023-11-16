#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


#LOADING NECESSARY PACKAGES
import numpy as np
import pandas as pd 
from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules


# In[3]:


dataset = pd.read_excel(r"C:\Users\AMEY DESHMUKH\OneDrive\Desktop\DATA MINING\Market Basket Analysis\archive (1)\data_mining_project.xlsx")


# In[7]:


dataset.head()


# In[11]:


dataset['Itemname'] = dataset['Itemname'].str.strip() 
dataset.dropna(axis=0, subset=['BillNo'], inplace=True)
dataset['BillNo'] = dataset['BillNo'].astype('str')
dataset.head()


# In[12]:


dataset['Country'].value_counts()


# In[13]:


dataset.shape


# In[43]:


#SEPERATING TRANSACTIONS FOR GERMANY
mybasket = (dataset[dataset['Country'] == "Germany"]
            .groupby(['BillNo', 'Itemname']) 
            ['Quantity'].sum()
            .unstack()
            .reset_index()
            .fillna(0)
            .set_index('BillNo'))


# In[44]:


#VIEWING TRANSACTION BASKET FOR GERMANY
mybasket.head()


# In[ ]:




