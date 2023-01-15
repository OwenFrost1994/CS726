import globals
import math
import numpy as np
import scipy

globals.initialize()

def objg(x,mode):
    z = ()
    gamma = 100.0
    m = len(x)
    if  m%2:
        raise ValueError('n must be multiple of 2')
    else:
        n = int(m/2)
    scale = 10
    f = 0
    g = np.zeros((m,))
    H = gamma*np.ones((m,m))
    # Do dense first
    for i in range(n):
        k = 2*i
        j = 2*i+1
        v= x[j] - x[k]**2
        f += scale*v**2 + (1-x[k])**2
        g[k] = -4*scale*x[k]*v + 2*x[k] - 2
        g[j] = 2*scale*v
        # this is just sparse 2 by 2 block
        H[k,k] += 12*scale*x[k]**2-4*scale*x[j]+2
        H[k,j] += -4*scale*x[k]
        H[j,k] += -4*scale*x[k]
        H[j,j] += 2*scale
  
    if mode & 1:
        globals.numf += 1
        z += (f+gamma*(np.sum(x)-1)**2/2,)
      
    if mode & 2:
        globals.numg += 1
        g += gamma*(np.sum(x)-1)*np.ones((m,))
        z += (g,)
  
    if mode & 4:
        globals.numH += 1
        z += (H,)
        # z += (scipy.sparse.csc_matrix(H, dtype=float),)
      
    return z
