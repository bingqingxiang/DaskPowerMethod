
# coding: utf-8

# Using DASK covariance matrix for centralized power method 

# In[4]:


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
import dask
import dask.array as da


# In[5]:


def power_dask(data,x_init):
    A=da.matmul(data,da.transpose(data))
    A.compute()
    T=150
    y=x_init
    for t in range(T):
        v=np.matmul(A,y)
        y=v/np.linalg.norm(v)


# In[6]:


def power(A,x_init):
    T=150
    y = x_init
    for t in range (T):
        y= np.matmul(A,y)
        y=y/LA.norm(y)


# In[7]:


#Data generation
DataDimension=100
TotalSamples=10e7
NumberOfNodes=20
mu, sigma = 0, 1 # mean and standard deviation
DataSamples = np.random.normal(mu, sigma, (int(DataDimension), int(TotalSamples)))


# In[8]:


N=TotalSamples
dim=DataDimension
k=1
data = DataSamples
x_init = np.random.randn(dim,k)


# In[9]:



#Implement centro PCA without dask
CovarianceMatrix = np.matmul(DataSamples, DataSamples.transpose())/TotalSamples
power=power(CovarianceMatrix,x_init)


# In[10]:



# power method using dask
n=NumberOfNodes # number of chunks
chunk_size=int(N/n)
data_d = da.from_array(data, chunks=(dim,chunk_size,))

