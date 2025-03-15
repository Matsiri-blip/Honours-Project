#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# ## Create a random adjacent matrix

# In[284]:


##creating a random matrix
n = 4
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

# In[269]:


x2 =np.array([0,1,2])
y2 = np.array([0,2,0])
##conditions for diagonal lines
x12 = np.array([0,1])
y12 = np.array([0,2])
x23 = np.array([1,2])
y23 = np.array([2,0])


# In[270]:


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


# ## N = 4

# In[323]:


x3 =np.array([0,0,1,1])
y3 = np.array([0,1,1,0])
##conditions for diagonal lines
x23 = np.array([0,1])
y23 = np.array([0,1])
x14 = np.array([0,1])
y14 = np.array([1,0])
##define the edges
## to save time so that each plot is recalled later
def plot_12():
    return plt.axhline(y = 1, xmin = 0.05, xmax = 0.95)
def plot_13():
    return plt.axvline(x = 0, ymin = 0.05, ymax = 0.95) 
def plot_14():
    return plt.plot(x14,y14)
def plot_23():
    return plt.plot(x23,y23)
def plot_24():
    return plt.axvline(x = 1, ymin = 0.05, ymax = 0.95) 
def plot_34():
    return plt.axhline(y = 0, xmin = 0.05, xmax = 0.95)


# In[325]:


plt.scatter(x3,y3)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.01,0.05,'3')
plt.text(0.99,0.05,'4')
plt.axis('off')
plt.text(0.5,0.5,'0',fontsize = 15)
plt.title('GRAPH OF 3 VERTEX')
plt.show()
binary_matrix(4,0)


# In[326]:


plt.scatter(x3,y3)
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.01,0.05,'3')
plt.text(0.99,0.05,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,1)


# In[335]:


plt.scatter(x3,y3)
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')

plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,2)


# In[337]:


plt.scatter(x3,y3)
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,3)


# In[340]:


plt.scatter(x3,y3)
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')

plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,4)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]


# In[342]:


plt.scatter(x3,y3)
plot_23()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')

plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,5)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]


# In[347]:


plt.scatter(x3,y3)
plot_23()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,6)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[349]:


plt.scatter(x3,y3)
plot_23()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,7)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[351]:


plt.scatter(x3,y3)
plot_14()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,8)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[353]:


plt.scatter(x3,y3)
plot_34()
plot_14()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,9)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[356]:


plt.scatter(x3,y3)
plot_14()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,10)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[362]:


plt.scatter(x3,y3)
plot_14()
plot_34()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,11)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[364]:


plt.scatter(x3,y3)
plot_14()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,12)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[366]:


plt.scatter(x3,y3)
plot_14()
plot_34()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,13)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[368]:


plt.scatter(x3,y3)
plot_14()
plot_23()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,14)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[370]:


plt.scatter(x3,y3)
plot_14()
plot_23()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,15)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[371]:


plt.scatter(x3,y3)
plot_13()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,16)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[373]:


plt.scatter(x3,y3)
plot_13()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,17)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[375]:


plt.scatter(x3,y3)
plot_13()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,18)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[378]:


plt.scatter(x3,y3)
plot_13()
plot_34()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,19)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[379]:


plt.scatter(x3,y3)
plot_13()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,20)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[380]:


plt.scatter(x3,y3)
plot_13()
plot_23()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,21)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[382]:


plt.scatter(x3,y3)
plot_13()
plot_23()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,22)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[383]:


plt.scatter(x3,y3)
plot_13()
plot_23()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,23)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[384]:


plt.scatter(x3,y3)
plot_13()
plot_14()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,24)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[389]:


plt.scatter(x3,y3)
plot_13()
plot_14()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,25)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[394]:


plt.scatter(x3,y3)
plot_13()
plot_14()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,26)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[395]:


plt.scatter(x3,y3)
plot_13()
plot_14()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,27)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[397]:


plt.scatter(x3,y3)
plot_13()
plot_14()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,28)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[401]:


plt.scatter(x3,y3)
plot_13()
plot_14()
plot_23()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,29)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[402]:


plt.scatter(x3,y3)
plot_13()
plot_14()
plot_23()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,30)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[403]:


plt.scatter(x3,y3)
plot_13()
plot_14()
plot_23()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,31)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[404]:


plt.scatter(x3,y3)
plot_12()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,32)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[405]:


plt.scatter(x3,y3)
plot_12()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,33)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[407]:


plt.scatter(x3,y3)
plot_12()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,34)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[408]:


plt.scatter(x3,y3)
plot_12()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,35)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[409]:


plt.scatter(x3,y3)
plot_12()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,36)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[411]:


plt.scatter(x3,y3)
plot_12()
plot_34()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,37)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[413]:


plt.scatter(x3,y3)
plot_12()
plot_24()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,38)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[415]:


plt.scatter(x3,y3)
plot_12()
plot_23()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,39)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[417]:


plt.scatter(x3,y3)
plot_12()
plot_14()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,40)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[418]:


plt.scatter(x3,y3)
plot_12()
plot_14()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,41)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[419]:


plt.scatter(x3,y3)
plot_12()
plot_14()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,42)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[421]:


plt.scatter(x3,y3)
plot_12()
plot_14()
plot_34()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,43)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[424]:


plt.scatter(x3,y3)
plot_12()
plot_14()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,44)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[426]:


plt.scatter(x3,y3)
plot_12()
plot_14()
plot_23()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,45)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[430]:


plt.scatter(x3,y3)
plot_12()
plot_14()
plot_23()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,46)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[431]:


plt.scatter(x3,y3)
plot_12()
plot_14()
plot_23()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,47)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[433]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,48)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[434]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,49)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[435]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,50)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[436]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_34()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,51)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[438]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,52)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[439]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_23()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,53)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[441]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_23()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,54)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[442]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_23()
plot_34()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,55)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[444]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,56)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[445]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,57)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[448]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,58)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[449]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,59)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[451]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,60)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[453]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plot_23()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,61)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[456]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plot_23()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,62)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[457]:


plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plot_23()
plot_34()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
binary_matrix(4,63)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], but the code has to be ran first


# In[ ]:




