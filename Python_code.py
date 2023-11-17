#!/usr/bin/env python
# coding: utf-8

# In[1]:


#INSTALL mlxtend
get_ipython().system('pip install mlxtend')


# In[2]:


#LOADING NECESSARY PACKAGES
import numpy as np
import pandas as pd 
from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules


# In[3]:


#IMPORT DATASET 
dataset = pd.read_excel(r"C:\Users\AMEY DESHMUKH\OneDrive\Desktop\DATA MINING\Market Basket Analysis\archive (1)\dataset1.xlsx")


# In[5]:


#VIEW FIRST 5 ROWS OF THE DATASET 
dataset.head()


# In[6]:


#DATA CLEANINIG 
dataset['Itemname'] = dataset['Itemname'].str.strip() 
dataset.dropna(axis=0, subset=['BillNo'], inplace=True)
dataset['BillNo'] = dataset['BillNo'].astype('str')
dataset.head()


# In[7]:


#COUNTING TOTAL RECORDS FOR EACH COUNTRY 
dataset['Country'].value_counts()


# In[8]:


#CHECK TOTAL COLUMNS AND ROWS IN THE DATASET 
dataset.shape


# In[9]:


#SEPERATING TRANSACTIONS FOR GERMANY
mybasket = (dataset[dataset['Country'] == "Germany"]
            .groupby(['BillNo', 'Itemname']) 
            ['Quantity'].sum()
            .unstack()
            .reset_index()
            .fillna(0)
            .set_index('BillNo'))


# In[10]:


#VIEWING TRANSACTION BASKET FOR GERMANY
mybasket.head()


# In[11]:


#CONVERTING ALL POSITIVE VALUES TO 1 AND OTHERS TO 0
def my_encode_units (x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
 
mybasketsets = mybasket.applymap(my_encode_units)


# In[22]:


#FREQUENT ITEMSETS
frequent_itemsets = apriori(mybasketsets, min_support = 0.07, use_colnames = True)


# In[13]:


#GENERATING RULES
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)


# In[26]:


#TOP 100 RULES
rules.head(30)


# In[21]:


#RULES BASED ON CONDITIONS
rules[(rules['lift'] >=3) & (rules['confidence'] >=0.7)]


# In[ ]:




