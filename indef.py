import globals
import math
import numpy as np
import scipy.sparse as sp

def indef(x,mode):
  z = ()
  N = len(x)
  alpha = 0.5
  scale = 1e-6
  if mode & 1:
    globals.numf += 1
    f = np.sum(x) + scale*np.dot(x,x)
    for i in range(1,N-1):
      f += alpha*math.cos(2*x[i]-x[0]-x[N-1])
    z += (f,)
      
  if mode & 2:
    globals.numg += 1
    g = np.ones((N,),dtype=float)+2*scale*x
    for k in range(1,N-1):
      g[k] -= 2.0*alpha*math.sin(2*x[k]-x[0]-x[N-1])
      g[0] += alpha*math.sin(2*x[k]-x[0]-x[N-1])
      g[N-1] += alpha*math.sin(2*x[k]-x[0]-x[N-1])
    z += (g,)
      
  if mode & 4:
    globals.numH += 1
    row = np.zeros((5*N,),dtype=int)
    col = np.zeros((5*N,),dtype=int)
    val = np.zeros((5*N,),dtype=float)
    for k in range(0,N):
      row[k] = k
      col[k] = k
      val[k] = 2*scale
      row[k+N] = k
      col[k+N] = 0
      row[k+2*N] = 0
      col[k+2*N] = k
      row[k+3*N] = k
      col[k+3*N] = N-1
      row[k+4*N] = N-1
      col[k+4*N] = k
    for k in range(1,N-1):
      t1 = alpha*math.cos(2*x[k]-x[0]-x[N-1])
      val[0] -= t1
      val[N-1] -= t1
      val[2*N-1] -= t1
      val[3*N-1] -= t1     
      val[k] -= 4*t1      
      val[k+N] = 2*t1
      val[k+2*N] = 2*t1
      val[k+3*N] = 2*t1
      val[k+4*N] = 2*t1  
    z += (sp.csc_matrix((val, (row,col)), shape=(N,N)),)
     
  return z
