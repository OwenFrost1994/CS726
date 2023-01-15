import math
import numpy as np
import globals 
from SteepDescent import SteepDescent
from obja import obja
from objb import objb
from objc import objc
from objd import objd
from geodesic import geodesic

def objsimp(x,mode):
  z = ()
  if mode & 1:
    globals.numf += 1
    z += (x**2 + math.exp(x),)
  if mode & 2:
    globals.numg += 1
    z = z + (2*x + math.exp(x),)
  if mode & 4:
    globals.numH += 1
    z = z + (2 + math.exp(x),)
  return z

def probres(inform,x,params):
  np.set_printoptions(formatter={'float': '{:8.4f}'.format})
  if inform['status'] == 0:
    print('CONVERGENCE FAILURE:')
    print('{0} steps were taken without gradient size decreasing below {1:.4g}.\n'.format(inform['iter'],params['toler']))
  else:
    print('Success: {0} steps taken\n'.format(inform['iter']))

  print('Ending point: {0}'.format(x['p']))
  print('Ending value: {0:.4g}'.format(x['f']))
  print('No. function evaluations: {0}'.format(globals.numf))
  print('Ending gradient: {0}'.format(x['g']))
  print('No. gradient evaluations {0}'.format(globals.numg))
  print('Norm of ending gradient: {0:8.4g}\n'.format(np.linalg.norm(x['g'])))

globals.initialize()
#[a,c] = geodesic(np.ones((10,),dtype=float),5)
[a,c] = geodesic(np.arange(1,11,dtype=float),5)

sdparams = {'maxit': 1000,'toler': 1.0e-4}
globals.initialize()
x = {'p': 1}
print('Problem simple');
[inform,xnew] = SteepDescent(objsimp,x,sdparams)
probres(inform,xnew,sdparams)

globals.initialize()
x = {'p': [-1.2, 1.0]}
print('Problem obja');
[inform,xnew] = SteepDescent(obja,x,sdparams)
probres(inform,xnew,sdparams)

globals.initialize()
x = {'p': [-1.2, 1.0]}
print('Problem objb');
[inform,xnew] = SteepDescent(objb,x,sdparams)
probres(inform,xnew,sdparams)

globals.initialize()
x = {'p': [-1.2, 1.0]}
print('Problem objc');
[inform,xnew] = SteepDescent(objc,x,sdparams)
probres(inform,xnew,sdparams)

sdparams = {'maxit': 1000,'toler': 3.0e-3}
globals.initialize()
x = {'p': [-1.2, 1.0]}
print('Problem objd');
[inform,xnew] = SteepDescent(objd,x,sdparams)
probres(inform,xnew,sdparams)

