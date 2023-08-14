#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Cleaning and Preparing The data for Model Training
#Dataset Link:https://www.kaggle.com/datasets/sdolezel/black-friday
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


#importing the dataset
df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')


# In[ ]:


Problem Statement
A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) 
against various products of different categories. They have shared purchase summary of various customers for selected high
volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, 
stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.

Now, they want to build a model to predict the purchase amount of customer against various products which will help them 
to create personalized offer for customers against different products.


# In[8]:


df_train.head()


# In[9]:


df_test.head()


# In[ ]:


df=pd.merge(df_train,df_test, on='User_ID', how = 'right')


# In[13]:


df=df_train.append(df_test)
df.head()


# In[14]:


df.info()


# In[15]:


df.describe()


# In[ ]:



df.drop(['User_ID'],axis=1,inplace= True ) 


# In[98]:


df.head()


# In[105]:


#Handling the Categorical variables 
df['Gender']=df['Gender'].map({'M':0,'F':0})
df.head()


# In[100]:


df['Age'].unique()


# In[23]:


df['Age']=df['Age'].map({'0-17':0,'18-25':1,'26-35':2,'36-45':3,'46-50':4,'51-55':5,'55+':6})


# In[25]:


df.head(10)


# In[ ]:





# In[28]:


#second technique for transforming the categorical variables into numerical
from sklearn import preprocessing

#label_encoder object knows how to understand word labels 
label_encoder=preprocessing.LabelEncoder()

#Encode labels in column 'species'
df['Age']=label_encoder.fit_transform(df['Age'])

df['Age'].unique()


# In[29]:


df.head()


# In[30]:


#second technique for transforming the categorical variables into numerical
from sklearn import preprocessing

#label_encoder object knows how to understand word labels 
label_encoder=preprocessing.LabelEncoder()

#Encode labels in column 'species'
df['Gender']=label_encoder.fit_transform(df['Gender'])

df['Gender'].unique()


# In[31]:


df.head()


# In[32]:


##fixing ca 
df_city=pd.get_dummies(df['City_Category'],drop_first=True)


# In[35]:


df1= pd.concat([df,df_city],axis=1)
df.head()


# In[129]:


df.drop(['City_Category',axis=1,inplace=True])


# In[38]:


df.head()


# In[39]:


#Misssng values
df.isnull().sum()


# In[ ]:


#since we're just gonna use product_category 2 and 3 for our analysis, and purchase is gonna be used in test df so we're just gonna fill
the 1 and 2s for the df 


# In[40]:


##focus on replacing missing values
df['Product_Category_1'].unique()


# In[41]:


df['Product_Category_1'].unique()


# In[45]:


df['Product_Category_2'].value_counts()


# In[ ]:


#now we're filling the missing value w mode as the range for the value 
#is discrete so best way is mode value


# In[62]:


# Replace the missing value with Mode


# In[66]:


df['Product_Category_2'].mode()[0]


# In[69]:


df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2']).mode()[0]


# In[ ]:





# In[70]:


df['Product_Category_2'].isnull().sum()


# In[ ]:


#missing value


# In[72]:


df['Product_Category_2'].mode()[0]


# In[73]:


df['Product_Category_2'].mode()[0]


# In[75]:


df['Product_Category_2'].mode()[0]


# In[76]:


df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2']).mode()[0]


# In[77]:


df['Product_Category_2'].isnull().sum()


# In[79]:


#PRODUCT_CATEGORY_3 REPLACING MISSING VALUES
df['Product_Category_3'].mode()[0]


# In[80]:


df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3']).mode()[0]


# In[81]:


df['Product_Category_3'].isnull().sum()


# In[82]:


df.head()


# In[83]:


df['Stay_In_Current_City_Years'].unique()


# In[ ]:


#now replacing the 4+ with 4 in the value 


# In[ ]:


df['Stay_In_Current_City_Years']= df['Stay_In_Current_City_Years'].str.replace('+',' ')


# In[86]:


df.head()


# In[87]:


df.info()


# In[ ]:


#here after checking all the varibales, stay_in _current year is an object
#but has numerical values in it


# In[ ]:


##converting object into integer


# In[89]:


df['Stay_In_Current_City_Years']= df['Stay_In_Current_City_Years'].astype(int)


# In[90]:


df.info()


# In[ ]:


#VISUALIZATION 
sns.barplot('Age','Purchase',hue='Gender',data=df)


# In[ ]:


#VISUALIZATION 
sns.barplot('Age','Purchase',hue='Gender',data=df)


# In[ ]:


#PURCHASING OF MEN OVER WOEM IS HIGH HAHAHAHAHHA


# In[1]:


#visualiztion of Purchase with Occupation
import seaborn as sns


# In[ ]:


sns.barplot(x='Product_Category_1',y='Purchase',hue='Gender',data=df)


# In[ ]:





# In[ ]:





# In[113]:


df.head()


# In[115]:


##feature scaling 
df_test=df[df['Purchase'].isnull()]


# In[122]:


df_train=df[~df['Purchase'].isnull()]
X=df_train[:-1]
X.head()


# In[125]:


y=df['Purchase']


# In[126]:


y


# In[119]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(
y,test_size=0.33,random_state=42)


# In[127]:


##feature scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




