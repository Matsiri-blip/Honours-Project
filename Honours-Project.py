#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# ## Create a random adjacent matrix

# In[210]:


##creating a random matrix
n = 3
a  = np.random.randint(2,size = (n,n))
np.random.seed(42)
a


# In[76]:


##ensure that the matrix is symmetric 
for i in range(n):
    for j in range(n):
        a[i,j]=a[j,i]

##diagonals are 0
np.fill_diagonal(a,0)
a


# In[79]:


a[1][0:]


# In[65]:


a[1][2:]=np.array([0,1])
a


# ## IDEA: Turn a number into binary, then turn the binary into a vector"

# In[188]:


##here i define the number of elements in the upper triangular matrix
def binary_matrix(n,k):
    m = (n**2-n)/2
    binary = int(bin(k)[2:])
    binary_vec = [int(d) for d in str(binary)]
    l = len(binary_vec)
    if m!=len(binary_vec): 
        horizontal_matrix = np.concatenate((np.zeros(int(m-l)),binary_vec))
    else:
        horizontal_matrix = np.array([int(i) for i in str(binary)])
    ##here this returns the binary matrix
    return horizontal_matrix


# # want to turn this horizontal vec into an adjacent matrix

# In[72]:


horizontal_matrix


# In[73]:


horizontal_matrix[n-1:n+1]


# In[80]:


a[0][1:]=horizontal_matrix[0:n-1]
a[1][2:]=horizontal_matrix[n-1:n+1]
a[2][3:]=horizontal_matrix[n+1:n+2]
for i in range(n):
    for j in range(n):
        a[j,i]=a[i,j]
a


# In[75]:


horizontal_matrix[n+1:n+2]


# In[94]:


# import matplotlib for drawing graphs

import matplotlib.pyplot as plt


# In[189]:


#putting the vertex at (0,0)

x0 = np.array([0])
y0 = np.array([0])


# In[104]:


plt.plot(x0,y0,'o')
plt.axis('off')
plt.title('GRAPH OF 1 VERTEX')
plt.show()


# In[97]:


##for n = 2


# In[192]:


x1 =np.array([0,1])
y1 = np.array([0,0])


# In[193]:


plt.scatter(x1,y1)
plt.text(-0.01,0.01,'1')
plt.text(0.99,0.01,'2')
plt.text(0.5,-0.01,'0',fontsize = 15)
plt.axis('off')
plt.title('GRAPH OF 2 VERTEX')
plt.show()
binary_matrix(2,0)


# In[196]:


plt.title('GRAPH OF 2 VERTEX')
plt.scatter(x1,y1)
plt.text(-0.01,0.01,'1')
plt.text(0.99,0.01,'2')
plt.axhline(y = 0, xmin = 0.05, xmax = 0.95) 
plt.text(0.5,-0.01,'1',fontsize = 15)
plt.axis('off')
plt.show()
binary_matrix(2,1)


# # N = 3

# In[241]:


x2 =np.array([0,1,2])
y2 = np.array([0,2,0])
##conditions for diagonal lines
x12 = np.array([0,1])
y12 = np.array([0,2])
x23 = np.array([1,2])
y23 = np.array([2,0])


# In[228]:


plt.scatter(x2,y2)
plt.text(-0.01,0.01,'1')
plt.text(0.99,2.05,'2')
plt.text(1.99,0.05,'3')
plt.axis('off')
plt.text(1,1,'0',fontsize = 15)
plt.title('GRAPH OF 3 VERTEX')
plt.show()
binary_matrix(3,0)


# In[242]:


plt.axhline(y = 0, xmin = 0.05, xmax = 0.95) ###between vertices 1,3
plt.scatter(x2,y2)
plt.text(-0.01,0.01,'1')
plt.text(0.99,2.05,'2')
plt.text(1.98,0.05,'3')
plt.text(1,1,'1',fontsize = 15)
plt.axis('off')
plt.title('GRAPH OF 3 VERTEX')
plt.show()
binary_matrix(3,1)


# In[253]:


plt.plot(x23,y23)##edge between 2&3
plt.scatter(x2,y2)
plt.text(-0.01,0.01,'1')
plt.text(1,1,'2',fontsize = 15)
plt.text(0.99,2.05,'2')
plt.text(1.99,0.05,'3')
plt.axis('off')
plt.title('GRAPH OF 3 VERTEX')
plt.show()
binary_matrix(3,2)


# In[254]:


plt.plot(x23,y23)##edge between 2&3 
plt.axhline(y = 0, xmin = 0.05, xmax = 0.95)##and also 1&3
plt.scatter(x2,y2)
plt.text(1,1,'3',fontsize = 15)
plt.text(-0.01,0.01,'1')
plt.text(0.99,2.05,'2')
plt.text(1.99,0.05,'3')
plt.axis('off')
plt.title('GRAPH OF 3 VERTEX')
plt.show()
binary_matrix(3,3)


# In[255]:


plt.plot(x12,y12)##edge between 1,2
plt.scatter(x2,y2)
plt.text(-0.01,0.01,'1')
plt.text(0.99,2.05,'2')
plt.text(1.99,0.05,'3')
plt.text(1,1,'4',fontsize = 15)
plt.axis('off')
plt.title('GRAPH OF 3 VERTEX')
plt.show()
binary_matrix(3,4)


# In[262]:


plt.plot(x12,y12)##edge between 1,2
plt.plot(x23,y23)##edge between 2&3 
plt.scatter(x2,y2)
plt.text(-0.01,0.01,'1')
plt.text(0.99,2.05,'2')
plt.text(1,1,'5',fontsize = 15)
plt.text(1.99,0.05,'3')
plt.axis('off')
plt.title('GRAPH OF 3 VERTEX')
plt.show()
binary_matrix(3,5)


# In[258]:


plt.plot(x12,y12)##edge between 1,2
plt.axhline(y = 0, xmin = 0.05, xmax = 0.95)##and also 1&3
plt.scatter(x2,y2)
plt.text(-0.01,0.01,'1')
plt.text(0.99,2.05,'2')
plt.text(1,1,'6',fontsize = 15)
plt.text(1.99,0.05,'3')
plt.axis('off')
plt.title('GRAPH OF 3 VERTEX')
plt.show()
binary_matrix(3,6)


# In[264]:


plt.plot(x12,y12)##edge between 1,2
plt.axhline(y = 0, xmin = 0.05, xmax = 0.95)##and also 1&3
plt.plot(x23,y23)##edge between 2&3 
plt.scatter(x2,y2)
plt.text(-0.01,0.01,'1')
plt.text(0.99,2.05,'2')
plt.text(1,1,'7',fontsize = 15)
plt.text(1.99,0.05,'3')
plt.axis('off')
plt.title('GRAPH OF 3 VERTEX')
plt.show()
binary_matrix(3,7)


# ## N = 3

# In[ ]:




