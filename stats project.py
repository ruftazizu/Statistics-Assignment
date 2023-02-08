#!/usr/bin/env python
# coding: utf-8

# ### import libraries
# 
# 

# In[32]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Load data into a pandas dataframe

# In[10]:


data = pd.read_csv("advertising.csv")
data


# In[12]:


#see rows and columns in the data
data.shape


# In[13]:


data.describe()


# In[14]:


#explore data
data.info()


# Check for missing values
# 

# In[16]:


missing_data = data.isnull()
missing_data.head()


# In[17]:


#let's print all the column names in the missing data and then whe can count and print missing data count
for column in list(missing_data.columns):
    #print all the column names
    print(column)
    #count the data values
    print(missing_data[column].value_counts())


# # As we saw the results of the codes in the upper part it is sure that this data is clean does not contain any missing values

# #### 1 Exploring the relathionships using plots
# 

# In[18]:


#Analysis between the individual campaign Spends and the Sales
#Will use matplotlib plotting the relationship
plt.scatter(x=data['TV'], y=data ['Sales'])
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()


# ### The plot between the TV campaign and Sales is showing that they both have a linear relashionship

# In[19]:


#lets see the relathionship between Radio campaign and sales
plt.scatter(x=data['Radio'], y=data['Sales'])
plt.xlabel('Radio')
plt.ylabel('Sales')
plt.show


# ### The plot between the Radio and Sales is showing that they have a non-linear relathionship

# In[21]:


#lets see the relathionship between Newspaer and Sales
#Using matplotlib plotting the relathionships
plt.scatter(x=data['Newspaper'], y=data['Sales'])
plt.xlabel('Newspaper')
plt.ylabel('Sales')
plt.show()


# ### The plot between the Newspaper and Sales is showing that they have a non-linear relathionship

# ### 2 Exploring retationships strenght using numbers - correlation

# In[22]:


#Lets do a correlation matrix between tv and sales variables
np.corrcoef(data['TV'], data['Sales'])


# #### As the result shows there is a very strong relationship between Tv and Sales campaign

# In[24]:


#Lets do a correlation matrix between Radio and Sales variables
np.corrcoef(data['Radio'], data['Sales'])


# #### As the result shows there is a very weak relationship between Radio and Sales campaign

# In[25]:


#Lets do a correlation matrix between Newspaper and Sales variables
np.corrcoef(data['Newspaper'], data['Sales'])


# #### As the result shows there is a very weak relationship between Newspaper and Sales campaign

# In[26]:


#Lets see the correlation of all variables in a matrix through pandas
data.corr()


# In[29]:


#Using the Seaborn module to plot the correlation between all the variables
sns.heatmap(data.corr(), cmap='coolwarm', annot=True)
plt.show()


# In[48]:


sns.pairplot(data)


# ### As its shown on the apper part the result shows, by looking at the correlation plot it is clear that Tv campaign and Sales are high in correlation on the other hand the Newspaper and Radion campaign have a week correlation with the Sales.

# In[50]:


#split dataset
X = data.drop('Sales', axis=1)
y = data['Sales']


# In[51]:


#import linear regression
from sklearn.linear_model import LinearRegression


# In[52]:


#create instance of LinearRegression
lm = LinearRegression()


# In[53]:


#fit the model with data
lm.fit(X,y)


# In[54]:


#get r-squared
lm.score


# In[56]:


#get coefficients
lm.coef_


# In[57]:


#get intercept
lm.intercept_


# In[58]:


#get p-values
import statsmodels.api as sm


# In[59]:



#create a fitted model with all features
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()


# In[60]:


#get p-values
est2.summary()


# ### The analysis and results suggest that there is a correlation between the individual campaigns and the sales figures. The radio and newspaper campaigns have a weaker correlation than the TV campaign and the sales figures. This indicates that spending money on a TV advertising campaign is more likely to result in a direct increase in sales. However, this does not necessarily mean that there is a causation between the two variables. Further analysis through A/B testing or other methods is necessary to determine the causal relationship.

# In[ ]:




