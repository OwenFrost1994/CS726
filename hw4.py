#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import array
import math


# In[2]:


def obja(x,mode):
    global numf, numg, numH
    z = ()
    if mode == 1:
        numf += 1
        z += (x[0]**2 + 5*(x[1]**2) + x[0] - 5*x[1],)
    if mode == 2:
        numg += 1
        z = z + (array([2*x[0] + 1, 10*x[1] - 5]),)
    if mode == 3:
        numH += 1
        z = z + (array([[2, 0], [0, 10]]),)
    return z


# In[3]:


numf = 0; numg = 0; numH = 0

f = obja(array([1, 2]),3)

print(f, numf, numg, numH)


# In[4]:


f = obja(array([1, 2]),1)

print(f, numf, numg, numH)


# In[5]:


def obja(x,mode):
    global numf, numg, numH
    z = ()
    if mode == 1:
        numf += 1
        z += (math.cos(x[0]) + math.exp(-x[0]),)
    if mode == 2:
        numg += 1
        z = z + (array([-math.sin(x[0]) - math.exp(-x[0])]),)
    if mode == 3:
        numH += 1
        z = z + (array([-math.cos(x[0]) + math.exp(-x[0])]),)
    return z


# In[6]:


numf = 0; numg = 0; numH = 0

f = obja(array([2]),3)

print(f, numf, numg, numH)


# In[7]:


f1 = obja(array([math.pi/2]),1)
print(f1, numf, numg, numH)
f2 = obja(array([math.pi/2]),2)
print(f2, numf, numg, numH)
f3 = obja(array([math.pi/2]),3)
print(f3, numf, numg, numH)


# In[8]:


f2[0][0]*f3[0][0]


# In[9]:


f = obja(array([3*math.pi/2]),1)
print(f, numf, numg, numH)
f = obja(array([3*math.pi/2]),2)
print(f, numf, numg, numH)
f = obja(array([3*math.pi/2]),3)
print(f, numf, numg, numH)


# In[10]:


from bisectS import bisect


# In[11]:


numf = 0; numg = 0; numH = 0
result = bisect(obja, math.pi/2, 3*math.pi/2, 1E-4)


# In[12]:


print('Interval is [%f, %f]' % (result[0], result[1]))
print('Number of iterations %f' % (result[2]))
print('Number of function evaluations used %f' % (numf + numg + numH))


# In[ ]:




