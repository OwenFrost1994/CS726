import globals
import numpy as np

globals.initialize()

def objc(x,mode):
    z = ()
    y = x[1] - x[0]**2
    
    if mode & 1:
        globals.numf += 1
        z += (100.*y**2 + (1-x[0])**2,)
    
    if mode & 2:
        globals.numg += 1
        z += (np.array([-400.*x[0]*y + 2*x[0] - 2., 200*y]),)
    
    if mode & 4:
        globals.numH += 1
        z += (np.array([[1200.*x[0]**2-400.*x[1]+2., -400.*x[0]], [-400.*x[0], 200.]]),)
    
    return z