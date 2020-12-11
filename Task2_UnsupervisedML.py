#!/usr/bin/env python
# coding: utf-8

# # Sparks Foundation Task2

#  The task is to predict the optimum number of clusters for Iris dataset.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ###### Loading Dataset

# In[4]:


iris = pd.read_csv('Datasets/iris.csv')
iris.head()


# ###### Drop unwanted column

# In[5]:


iris_data = iris.drop(['Id'], axis=1)
iris_data.head()


# In[6]:


iris_data.shape


# Now, This dataset has 150 rows and 5 columns

# ###### stores the Attributes/Predictors in variable x

# In[7]:


x = iris_data.iloc[:, [0,1,2,3]].values
x


# In[8]:


iris_data.info()


# In[9]:


iris_outcome = pd.crosstab(index=iris_data["Species"],  columns="count")      
iris_outcome


# ###### Exploratory Data Analysis

# In[10]:


sns.set()
sns.pairplot(iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']], hue="Species", 
             diag_kind="kde")


# In[11]:


iris_setosa=iris_data.loc[iris_data["Species"]=="Iris-setosa"]
iris_virginica=iris_data.loc[iris_data["Species"]=="Iris-virginica"]
iris_versicolor=iris_data.loc[iris_data["Species"]=="Iris-versicolor"]


# In[12]:


sns.FacetGrid(iris_data,hue="Species",size=3).map(sns.distplot,"SepalLengthCm").add_legend()
sns.FacetGrid(iris_data,hue="Species",size=3).map(sns.distplot,"SepalWidthCm").add_legend()
sns.FacetGrid(iris_data,hue="Species",size=3).map(sns.distplot,"PetalLengthCm").add_legend()
sns.FacetGrid(iris_data,hue="Species",size=3).map(sns.distplot,"PetalWidthCm").add_legend()
plt.show()


# ###### Data Modelling

# In[13]:


from sklearn.cluster import KMeans


# ###### Finding the optimum number of clusters for k-means classification

# In[14]:


wcss = []

for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)


# ###### Plotting the Elbow Curve

# In[15]:


#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.figure(figsize=(10,6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The elbow curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()


# The representation shows the elbow at the point 3, means we can choose k=3 clusters for the analysis.

# ###### Training and Prediction

# In[16]:


km = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
ykmeans = km.fit_predict(x)
ykmeans


# In[17]:


km.cluster_centers_[:, 0]


# In[18]:


km.cluster_centers_[:,1]


# ###### Visualising the clusters

# In[19]:


plt.figure(figsize=(6,6))
plt.scatter(x[ykmeans == 0, 0], x[ykmeans == 0, 1], s=100, c='yellow', marker='^', label='Iris-setosa')
plt.scatter(x[ykmeans == 1, 0], x[ykmeans == 1, 1], s=100, c='yellowgreen', marker='s', label='Iris-versicolour')
plt.scatter(x[ykmeans == 2, 0], x[ykmeans == 2, 1], s=100, c='lightblue', marker="o", label='Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids',
            marker='X')
plt.title("Iris Data optimum clusters", fontsize=15)
plt.legend()


# In[ ]:




