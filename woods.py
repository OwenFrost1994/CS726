import globals
import numpy as np
import scipy.sparse as sp

def woods(x,mode):
    z = ()
    N = len(x)
    if N % 4 != 0: 
        raise ValueError('must have number of variables mult of 4')
    else:
        M = int(N/4)
    if mode & 1:
        globals.numf += 1
        f = 0.
        for i in range(M):
            f += 100*(x[4*i+1]-x[4*i]**2)**2 + (1-x[4*i])**2 + 90*(x[4*i+3]-x[4*i+2]**2)**2 + (1-x[4*i+2])**2 + 10*(x[4*i+1]+x[4*i+3]-2)**2 + 0.1*(x[4*i+1]-x[4*i+3])**2
        z += (f,)
      
    if mode & 2:
        globals.numg += 1
        g = np.empty((N,),dtype=float)
        for k in range(0,int(N),4):
            t1 = x[k+1]-x[k]**2
            t2 = 20*(x[k+1]+x[k+3]-2)
            t3 = x[k+3]-x[k+2]**2
            t4 = 0.2*(x[k+1]-x[k+3])
            g[k] = -400*t1*x[k] - 2*(1-x[k])
            g[k+1] = 200*t1 + t2 + t4
            g[k+2] = -360*t3*x[k+2] -2*(1-x[k+2])
            g[k+3] = 180*t3 + t2 - t4
        z += (g,)
      
    if mode & 4:
        globals.numH += 1
        row = np.empty((10*M,),dtype=int)
        col = np.empty((10*M,),dtype=int)
        val = np.empty((10*M,),dtype=float)
        for k in range(0,int(N),4):
            j = int(2.5*k)
            ind = range(j,j+10)
            row[ind] = [k, k, k+1, k+1, k+1, k+2, k+2, k+3, k+3, k+3]
            col[ind] = [k, k+1, k, k+1, k+3, k+2, k+3, k+1, k+2, k+3]
            t1 = -400*x[k]
            t2 = -360*x[k+2]
            val[ind] = [-400*x[k+1]+1200*x[k]**2+2, t1, t1, 220.2, 19.8, -360*x[k+3]+1080*x[k+2]**2+2, t2, 19.8, t2, 200.2]
#      H = sp.csc_matrix((val, (row,col)), shape=(N,N))
#      z += (H.toarray(),)
        z += (sp.csc_matrix((val, (row,col)), shape=(N,N)),)
     
    return z
