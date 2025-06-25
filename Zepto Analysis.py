#!/usr/bin/env python
# coding: utf-8

# Definition :
# * This dataset provides a detailed view of the product catalog and pricing structure of Zepto, a fast-growing 10-minute grocery delivery platform. The data captures essential attributes for over 3,000+ SKUs (Stock Keeping Units) across various categories like Fruits & Vegetables, Dairy, Packaged Foods, Beverages, and more

# The data is structured to support various types of retail analysis, including:
# 
# * Discount trends by category
# * Inventory availability and stock-outs
# * Price distribution and pricing strategy
# * Product naming patterns (suitable for word cloud or NLP tasks)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px 

from wordcloud import wordcloud


# In[2]:


df=pd.read_csv('zepto_v2.csv',encoding="ISO-8859-1")
df.head()


# Data preprocessing 

# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.duplicated().sum()


# In[6]:


df.isna().sum()


# In[15]:


plt.figure(figsize=(12,10))
sns.countplot(x='Category',
             data=df,
             width=0.4,
             palette='dark:#5A9_r')
plt.title('Zepto food category')
plt.xticks(rotation='vertical');


# In[8]:


plt.figure(figsize=(12,6))
sns.countplot(x='availableQuantity',
              data=df,
              width=0.4,
             palette="blend:#7AB,#EDA")
plt.title('Zepto food quantity')
plt.xticks(rotation='horizontal');


# In[17]:


plt.figure(figsize=(8,4))
sns.countplot(x='outOfStock',
              data=df,
              palette='ch:s=.25,rot=-.25',
             width=0.4)
plt.legend()
plt.title('Zepto out of stock');


# In[10]:


df.head()


# In[18]:


Out_of_stock_items=df[df["outOfStock"]==True]
Out_of_stock_items[["name","quantity"]].sort_values(["quantity"],ascending=True)


# In[21]:


plt.figure(figsize=(12, 25))  # Tall figure
sns.barplot(data=Out_of_stock_items, y="name", x="quantity", dodge=False)

plt.title("Out-of-Stock Items and Their Quantities")
plt.xlabel("Quantity")
plt.ylabel("Item Name")
plt.tight_layout()
plt.show()


# In[27]:


d=df.select_dtypes(include='number')
for col in d:
    plt.figure(figsize=(15,7))
    sns.histplot(data=d,
    x=col,
    kde=True,
    bins=20,
    color='#7a5195')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=90)
    plt.show()


# In[31]:


df.head()


# In[33]:


Top_dicount_percentage=df.sort_values(["discountPercent"],ascending=False)
Top_dicount_percentage[["Category","name","mrp","discountPercent","discountedSellingPrice"]]


# In[39]:


plt.figure(figsize=(15,7))
sns.set_style('whitegrid')
sns.barplot(data=Top_dicount_percentage,
            x='Category',
            y='discountPercent',
            color='plum',
            edgecolor='black')

plt.xticks(rotation=60)
plt.title('Discount Percent Price')


# Qunatity selling

# In[40]:


Top_quantity=df.sort_values(['quantity'])
Top_quantity[['Category','quantity','availableQuantity']]


# In[42]:


plt.figure(figsize=(15,7))
sns.set_style('whitegrid')
sns.barplot(data=Top_quantity,
            x='Category',
            y='quantity',
            hue='availableQuantity',
           color='teal')
plt.xticks(rotation=60)
plt.title('Top quantity value');


# Correlation Map

# In[46]:


plt.figure(figsize=(15,7))
corr=df.select_dtypes(include="number").corr()
sns.heatmap(data=corr,
            annot=True,
            fmt='.2f')
plt.show()


# Applying in ANN model 

# Feature engineering

# In[47]:


df.info()


# In[54]:


label=['Category','name','outOfStock']
from sklearn.preprocessing import LabelEncoder


# In[55]:


le=LabelEncoder()


# In[56]:


for feature in label:
    df[feature]=le.fit_transform(df[feature])


# In[58]:


df.head()


# ANN Section ( Artificial intelligence network)

# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,classification_report,mean_absolute_error,mean_squared_error,r2_score


# In[61]:


import tensorflow as tf


# In[63]:


get_ipython().system(' pip install shap')


# In[64]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
import shap


# In[70]:


X = df.drop(df.columns[-2],axis=1)
y = df.iloc[:,-2]


# In[75]:


X_train,X_test,y_train,y_test=train_test_split(X,
                                               y,
                                               test_size=0.2,
                                               random_state=42)


# In[82]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[83]:


model=Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128,activation='relu'),
    Dense(64,activation='relu'),
    Dense(1)
])


# In[86]:


model.compile(optimizer=Adam(),
              loss='mean_squared_error',
              metrics=['mse'])
model.fit(X_train_scaled,
          y_train,epochs=100,
          batch_size=32,
          validation_split=0.1)


# In[87]:


y_pred = model.predict(X_test_scaled)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()


# In[88]:


print(classification_report(y_test,y_pred_labels))


# In[90]:


# Convert probabilities to class labels
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Convert probabilities to class labels
y_train_pred_labels = (y_train_pred > 0.5).astype(int)
y_test_pred_labels = (y_test_pred > 0.5).astype(int)

# Evaluate
from sklearn.metrics import classification_report

print("Train Report:")
print(classification_report(y_train, y_train_pred_labels))

print("Test Report:")
print(classification_report(y_test, y_test_pred_labels))


# Conclusion:
# 
# * Current ANN is severely underperforming and overfitting to the minority class.
# 
# * You should address class imbalance and retrain with class_weight, SMOTE, or both.
# 
# * After fixing, re-run the classification report to see improvements in both recall and precision across both classes.

# In[ ]:




