import globals
from numpy import array



def obja(x,mode):
    z = ()
    if mode & 1:
        globals.numf += 1
        z += (x[0]**2 + 5*x[1]**2 + x[0] - 5*x[1],)
    if mode & 2:
        globals.numg += 1
        z += (array([2*x[0] + 1, 10*x[1] - 5]),)
    if mode & 4:
        globals.numH += 1
        z += (array([[2, 0], [0, 10]]),)
    return z

# 
# [f,g] = objb(array([1, 2]),3)
# print(f,g)
