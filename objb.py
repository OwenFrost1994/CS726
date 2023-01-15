import globals
from numpy import array

globals.initialize()

def objb(x,mode):
    z = ()
    
    if mode & 1:
        globals.numf += 1
        z += (x[0]**2 + 5*x[0]*x[1] + 100*x[1]**2 - x[0] + 4*x[1],)
    
    if mode & 2:
        globals.numg += 1
        z += (array([2*x[0] + 5*x[1] - 1, 5*x[0] + 200*x[1] + 4]),)
        
    if mode & 4:
        globals.numH += 1
        z += (array([[2, 5], [5, 200]]),)
    
    return z