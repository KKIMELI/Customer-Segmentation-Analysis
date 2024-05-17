#Customer Segmentation Analysis
#1. Exploratory Data Analysis

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("C:/Users/Keith/Desktop/Data Analytics/Keith Portfolio Projects/Customer Segmentation Analysis/dataset analysis/Mall_Customers.csv")


# In[3]:


df.head()


# # Univariate Analysis

# In[4]:


df.describe()


# In[5]:


sns.distplot(df['Annual Income (k$)']);


# In[6]:


df.columns


# In[7]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# In[8]:


columns = ['Annual Income (k$)']
for i in columns:
    plt.figure()
    sns.kdeplot(data=df, x=i, shade=True, hue='Gender')


# In[9]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.kdeplot(data=df, x=i, shade=True, hue='Gender')


# In[10]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df, x='Gender', y=df[i])


# In[11]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis

# In[12]:


sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)')


# In[13]:


df=df.drop('CustomerID',axis=1)
sns.pairplot(df,hue='Gender')


# In[14]:


df.groupby(['Gender'])[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()


# In[15]:


specific_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
df_filtered = df[specific_cols]
df_filtered.corr()


# In[16]:


sns.heatmap(df_filtered.corr(), annot=True,cmap='coolwarm')


# # Clustering - Univariate, Bivariate, Multivariate

# In[17]:


clustering1 = KMeans(n_clusters=3)


# In[18]:


clustering1.fit(df[['Annual Income (k$)']])


# In[19]:


clustering1.labels_


# In[20]:


df['Income Cluster'] = clustering1.labels_
df.head()


# In[21]:


df['Income Cluster'].value_counts()


# In[22]:


clustering1.inertia_


# In[23]:


inertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)


# In[24]:


inertia_scores


# In[25]:


plt.plot(range(1,11),inertia_scores)


# In[26]:


df.columns


# In[27]:


df.groupby('Income Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()


# In[28]:


#Bivariate Clustering


# In[29]:


clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
df['Spending and Income Cluster'] = clustering2.labels_
df.head()


# In[30]:


inertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11),inertia_scores2)


# In[31]:


centers =pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']
centers


# In[32]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x ='Annual Income (k$)', y ='Spending Score (1-100)',hue='Spending and Income Cluster',palette='tab10')
plt.savefig('clustering_bivariate.png')


# In[33]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[34]:


df.groupby('Spending and Income Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()


# In[35]:


#multivariate clustering
from sklearn.preprocessing import StandardScaler


# In[36]:


scale = StandardScaler()


# In[37]:


df.head()


# In[38]:


dff = pd.get_dummies(df,drop_first=True)
dff.head()


# In[39]:


dff.columns


# In[40]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']]
dff.head()


# In[41]:


dff = scale.fit_transform(dff)


# In[42]:


dff = pd.DataFrame(scale.fit_transform(dff))
dff.head()


# In[43]:


inertia_scores3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    inertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11),inertia_scores3)


# In[44]:


df


# In[45]:


df.to_csv('Clustering.csv')


# In[ ]:




