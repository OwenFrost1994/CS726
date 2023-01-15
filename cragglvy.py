import globals
import math
import numpy as np
import scipy.sparse as sp

def cragglvy(x,mode):
  z = ()
  N = len(x)
  if N % 2 != 0: 
    raise ValueError('must have even number of variables')
  else:
    M = int((N-2)/2)
  if mode & 1:
    globals.numf += 1
    f = 0.
    for i in range(0,M):
      f += (math.exp(x[2*i])-x[2*i+1])**4+100*(x[2*i+1]-x[2*i+2])**6 + (math.tan(x[2*i+2]-x[2*i+3])+x[2*i+2]-x[2*i+3])**4 + x[2*i]**8+(x[2*i+3]-1)**2
    z += (f,)
      
  if mode & 2:
    globals.numg += 1
    g = np.zeros((N,),dtype=float)
    exk = math.exp(x[0])
    exkme3 = (exk-x[1])**3
    g[0] = 4*exkme3*exk + 8*x[0]**7
    xkme5 = 600*(x[1]-x[2])**5
    g[1] = -4*exkme3 + xkme5
    for k in range(2,N-3,2):
      exk = math.exp(x[k])
      exkme3 = (exk-x[k+1])**3
      xkkm1 = x[k]-x[k+1]
      cxkkm1 = math.cos(xkkm1)
      txkkm1 = math.tan(xkkm1)
      txkme3 = (txkkm1+xkkm1)**3*((1/cxkkm1)**2+1)
      g[k] = 4*exkme3*exk - xkme5 + 4*txkme3 + 8*x[k]**7
      xkme5 = 600*(x[k+1]-x[k+2])**5
      g[k+1] = -4*exkme3 + xkme5 - 4*txkme3 + 2*(x[k+1]-1)
    xkkm1 = x[N-2]-x[N-1]
    cxkkm1 = math.cos(xkkm1)
    txkkm1 = math.tan(xkkm1)
    txkme3 = (txkkm1+xkkm1)**3*((1/cxkkm1)**2+1)
    g[N-2] = -xkme5 + 4*txkme3
    g[N-1] = -4*txkme3 + 2*(x[N-1]-1)
    z += (g,)
      
  if mode & 4:
    globals.numH += 1
    row = np.zeros((3*N-2,),dtype=int)
    col = np.zeros((3*N-2,),dtype=int)
    val = np.zeros((3*N-2,),dtype=float)
    exk = math.exp(x[0])
    exkme3 = (exk-x[1])**3
    exkme2 = 12*(exk-x[1])**2
    for k in range(1,N):
      row[k] = k
      col[k] = k
      row[k+N-1] = k
      col[k+N-1] = k-1
      row[k+2*N-2] = k-1
      col[k+2*N-2] = k
    val[0] = 4*exkme3*exk + exkme2*exk**2 + 56*x[0]**6
    val[N] = -exkme2*exk
    gxkme4 = 3000*(x[1]-x[2])**4
    val[2*N-1] = val[N]
    val[1] = exkme2 + gxkme4
    val[N+1] = -gxkme4
    for k in range(2,N-3,2):
      exk = math.exp(x[k])
      exkme3 = (exk-x[k+1])**3
      exkme2 = 12*(exk-x[k+1])**2
      xkkm1 = x[k]-x[k+1]
      sxkkm1 = math.sin(xkkm1)
      cxkkm1 = math.cos(xkkm1)
      txkkm1 = math.tan(xkkm1)
      gtxkme = 4*(txkkm1 + xkkm1)**2*(6*cxkkm1 + 2*sxkkm1*txkkm1 + 3*cxkkm1*txkkm1**2 + 6*cxkkm1**3 + 2*sxkkm1*(xkkm1) + 3*cxkkm1**3*txkkm1**2)/cxkkm1**3
      val[k+2*N-2] = val[k+N-1]
      val[k] = 4*exkme3*exk + exkme2*exk**2 + gtxkme + gxkme4 + 56*x[k]**6
      val[k+N] = -exkme2*exk - gtxkme
      gxkme4 = 3000*(x[k+1]-x[k+2])**4
      val[k+1+2*N-2] = val[k+N]
      val[k+1] = exkme2 + gtxkme + gxkme4 + 2
      val[k+1+N] = -gxkme4
    val[N-2+2*N-2] = val[N-2+N-1]
    xkkm1 = x[N-2]-x[N-1]
    sxkkm1 = math.sin(xkkm1)
    cxkkm1 = math.cos(xkkm1)
    txkkm1 = math.tan(xkkm1)
    gtxkme = 4*(txkkm1 + xkkm1)**2*(6*cxkkm1 + 2*sxkkm1*txkkm1 + 3*cxkkm1*txkkm1**2 + 6*cxkkm1**3 + 2*sxkkm1*(xkkm1) + 3*cxkkm1**3*txkkm1**2)/cxkkm1**3
    val[N-2] = gtxkme + gxkme4
    val[N-2+N] = -gtxkme
    val[N-1+2*N-2] = val[N-2+N]
    val[N-1] = gtxkme + 2

    z += (sp.csc_matrix((val, (row,col)), shape=(N,N)),)
     
  return z

