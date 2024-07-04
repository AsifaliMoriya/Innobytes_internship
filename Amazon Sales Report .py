#!/usr/bin/env python
# coding: utf-8

# # Amazon Sales Analysis

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


cd C:\Users\asifa\OneDrive\Desktop\Amazon_Sales_Report


# In[3]:


df = pd.read_csv("Amazon Sale Report.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# ## Data Cleaning

# In[9]:


sales_data = df


# In[10]:


sales_data['Date'] = pd.to_datetime(sales_data['Date'])


# In[11]:


sales_data.drop(columns=['New', 'PendingS'], inplace=True)


# In[12]:


sales_data['ship-postal-code'].fillna(sales_data['ship-postal-code'].median(), inplace=True)

print(sales_data.isnull().sum())


# In[13]:


sales_data.dropna(subset=['Amount'], inplace=True)
sales_data.fillna({'ship-city': 'Unknown', 'ship-state': 'Unknown', 'ship-country': 'Unknown', 
                   'currency': 'Unknown', 'fulfilled-by': 'Unknown'}, inplace=True)


# ## Sales Overview

# In[14]:


overall_sales = sales_data['Amount'].sum()
print(f'Overall Sales: {overall_sales}')


# In[15]:


sales_trends = sales_data.groupby(sales_data['Date'].dt.to_period('M'))['Amount'].sum()
sales_trends.plot(kind='line', title='Sales Trends Over Time')
plt.ylabel('Sales Amount')
plt.xlabel('Date')
plt.show()


# In[16]:


sales_data['Weekday'] = sales_data['Date'].dt.day_name()
sales_by_weekday = sales_data.groupby('Weekday')['Amount'].sum().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.figure(figsize=(15, 8))
sales_by_weekday.plot(kind='bar', title='Sales Distribution by Weekday')
plt.ylabel('Sales Amount')
plt.xlabel('Weekday')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[17]:


sales_data['Month'] = sales_data['Date'].dt.month_name()
sales_data['Day'] = sales_data['Date'].dt.day
sales_pivot = sales_data.pivot_table(values='Amount', index='Day', columns='Month', aggfunc=np.sum, fill_value=0)
plt.figure(figsize=(15, 8))
sns.heatmap(sales_pivot, cmap='YlGnBu')
plt.title('Sales Heatmap by Month and Day')
plt.xlabel('Month')
plt.ylabel('Day')
plt.show()


# ## Product Analysis

# In[18]:


product_categories = sales_data['Category'].value_counts()
product_categories.plot(kind='bar', title='Product Category Distribution')
plt.ylabel('Count')
plt.xlabel('Product Category')
plt.show()


# In[19]:


category_avg_order_value = sales_data.groupby('Category')['Amount'].mean().sort_values(ascending=False)
plt.figure(figsize=(15, 8))
category_avg_order_value.plot(kind='bar', title='Category-wise Average Order Value')
plt.ylabel('Average Order Value')
plt.xlabel('Product Category')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[20]:


plt.figure(figsize=(15, 8))
corr = sales_data[['Qty', 'Amount']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[21]:


sizes_quantities = sales_data.groupby('Size')['Qty'].sum()
sizes_quantities.plot(kind='bar', title='Quantities Sold by Size')
plt.ylabel('Quantity Sold')
plt.xlabel('Size')
plt.show()


# ## Fulfillment Analysis

# In[22]:


fulfillment_methods = sales_data['Fulfilment'].value_counts()
fulfillment_methods.plot(kind='bar', title='Fulfillment Methods Distribution')
plt.ylabel('Count')
plt.xlabel('Fulfillment Method')
plt.show()


# In[23]:


fulfillment_status = sales_data.groupby(['Fulfilment', 'Status'])['Order ID'].count().unstack().fillna(0)
plt.figure(figsize=(15, 8))
fulfillment_status.plot(kind='bar', stacked=True)
plt.title('Fulfillment Method by Order Status')
plt.ylabel('Number of Orders')
plt.xlabel('Fulfillment Method')
plt.xticks(rotation=0, ha='right')
plt.legend(title='Order Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# ## Customer Segmentation

# In[24]:


customer_behavior = sales_data.groupby('Order ID')['Amount'].agg(['sum', 'count']).reset_index()
sns.scatterplot(x='count', y='sum', data=customer_behavior)
plt.title('Customer Segmentation by Buying Behavior')
plt.xlabel('Number of Purchases')
plt.ylabel('Total Amount Spent')
plt.show()


# In[25]:


customer_spending = sales_data.groupby('Order ID')['Amount'].sum()
plt.figure(figsize=(15, 8))
sns.histplot(customer_spending, bins=50, kde=True)
plt.title('Customer Spending Distribution')
plt.xlabel('Total Spending per Customer')
plt.ylabel('Frequency')
plt.show()


# In[26]:


customer_frequency = sales_data.groupby('Order ID')['Date'].count()
plt.figure(figsize=(15, 8))
sns.histplot(customer_frequency, bins=50, kde=True)
plt.title('Customer Purchase Frequency')
plt.xlabel('Number of Purchases per Customer')
plt.ylabel('Frequency')
plt.show()


# ## Geographical Analysis

# In[27]:


sales_by_state = sales_data.groupby('ship-state')['Amount'].sum().sort_values(ascending=False)
plt.figure(figsize=(20, 10)) 
sales_by_state.plot(kind='bar', title='Sales by State')
plt.ylabel('Sales Amount')
plt.xlabel('State')
plt.xticks(rotation=90, ha='right') 
plt.tight_layout()
plt.show()


# In[28]:


sales_by_city = sales_data.groupby('ship-city')['Amount'].sum().sort_values(ascending=False).head(30)
plt.figure(figsize=(20, 10))
sales_by_city.plot(kind='bar', title='Top 30 Sales by City')
plt.ylabel('Sales Amount')
plt.xlabel('City')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ## Business Insights

# In[29]:


print(f"Top Selling Product Categories:\n{product_categories.head()}")


# In[30]:


print(f"Top States by Sales Amount:\n{sales_by_state.head()}")


# In[31]:


print("Recommendations:")
print("1. Focus marketing efforts on top-selling product categories and states.")
print("2. Enhance fulfillment methods with higher effectiveness.")
print("3. Develop targeted campaigns for high-spending and frequent customers.")


# In[ ]:


from dataprep.eda import create_report


eda = sales_data
report = create_report(sales_data)
report.show_browser()

