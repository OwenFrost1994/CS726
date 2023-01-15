import timeit
import math
import numpy as np
import globals 
from Opt726plus import SteepDescent, BFGS, Newton, LBFGS, DogLeg, TNewton, cgTrust
from woods import woods
from indef import indef
from cragglvy import cragglvy
from objg import objg

def probres(inform,x,params):
  np.set_printoptions(formatter={'float': '{:8.4f}'.format})
  if inform['status'] == 0:
    print('CONVERGENCE FAILURE:')
    print('{0} steps were taken without gradient size decreasing below {1:.4g}.\n'.format(inform['iter'],params['toler']))
  else:
    print('Success: {0} steps taken\n'.format(inform['iter']))

  #print('Ending point: {0}'.format(x['p']))
  print('Ending value: {0:.4g}'.format(x['f']))
  print('No. function evaluations: {0}'.format(globals.numf))
  #print('Ending gradient: {0}'.format(x['g']))
  print('No. gradient evaluations {0}'.format(globals.numg))
  print('Norm of ending gradient: {0:8.4g}'.format(np.linalg.norm(x['g'])))
  print('No. Hessian evaluations {0}'.format(globals.numH))
  print('No. Factorizations {0}'.format(globals.numFact))
  print('Cg iterations {0}\n'.format(globals.cgits))

x = {'p': np.arange(1.0,21.0)}
globals.initialize()
[f,g,H] = woods(x['p'],7)

x = {'p': np.arange(1.0,2001.0)}
globals.initialize()
[H] = indef(x['p'],4)

x = {'p': np.arange(1.0,21.0)}
globals.initialize()
[f,g,H] = cragglvy(x['p'],7)

x = {'p': np.tile([-3.0,-1.0],500)} 
print('Steepest descent')
globals.initialize()
sdparams = {'maxit': 2000,'toler': 1.0e-4}
#[inform,path] = SteepDescent(woods,x,sdparams)
#probres(inform,path,sdparams)

print('Newton')
globals.initialize()
nparams = {'maxit': 100,'toler': 1.0e-4,'method': 'sppert'}
[inform,path] = Newton(woods,x,nparams)
probres(inform,path,sdparams)

print('BFGS')
globals.initialize()
nparams = {'maxit': 1000,'toler': 1.0e-4}
[inform,path] = BFGS(woods,x,nparams)
probres(inform,path,sdparams)

print('Newton')
x = {'p': np.arange(1.0,21.0)}
globals.initialize()
nparams = {'maxit': 100,'toler': 1.0e-4,'method': 'sppert'}
[inform,path] = Newton(objg,x,nparams)
probres(inform,path,sdparams)

print('BFGS')
x = {'p': np.arange(1.0,21.0)}
globals.initialize()
nparams = {'maxit': 1000,'toler': 1.0e-4}
[inform,path] = BFGS(objg,x,nparams)
probres(inform,path,sdparams)

print('LBFGS')
x = {'p': np.arange(1.0,21.0)}
globals.initialize()
nparams = {'maxit': 1000,'toler': 1.0e-4,'m': 3}
[inform,path] = LBFGS(objg,x,nparams)
probres(inform,path,sdparams)

print('LBFGS')
x = {'p': np.tile([-3.0,-1.0],500)}
globals.initialize()
nparams = {'maxit': 1000,'toler': 1.0e-4,'m': 17}
[inform,path] = LBFGS(woods,x,nparams)
probres(inform,path,sdparams)

print('DogLeg')
#x = {'p': np.array([-1.2, 1.0])}
x = {'p': np.arange(1.0,21.0)}
globals.initialize()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = DogLeg(objg,x,nparams)
probres(inform,path,sdparams)
 
print('DogLeg')
x = {'p': np.tile([-3.0,-1.0],500)}
globals.initialize()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = DogLeg(woods,x,nparams)
probres(inform,path,sdparams)

print('TNewton')
#x = {'p': np.array([-1.2, 1.0])}
x = {'p': np.arange(1.0,21.0)}
globals.initialize()
nparams = {'maxit': 1000,'toler': 1.0e-4}
[inform,path] = TNewton(objg,x,nparams)
probres(inform,path,sdparams)
 
print('TNewton')
x = {'p': np.tile([-3.0,-1.0],500)}
globals.initialize()
nparams = {'maxit': 1000,'toler': 1.0e-4}
[inform,path] = TNewton(woods,x,nparams)
probres(inform,path,sdparams)

print('cgTrust')
x# = {'p': np.array([-1.2, 1.0])}
x = {'p': np.arange(1.0,21.0)}
globals.initialize()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = cgTrust(objg,x,nparams)
probres(inform,path,sdparams)
 
print('cgTrust')
x = {'p': np.tile([-3.0,-1.0],500)}
globals.initialize()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = cgTrust(woods,x,nparams)
probres(inform,path,sdparams)

print('Large models')
print('woods:')

print('Newton')
x = {'p': np.tile([-3.0,-1.0],5000)}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'method': 'sppert'}
[inform,path] = Newton(woods,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('LBFGS')
x = {'p': np.tile([-3.0,-1.0],5000)}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'m': 17}
[inform,path] = LBFGS(woods,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('TNewton')
x = {'p': np.tile([-3.0,-1.0],5000)}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4}
[inform,path] = TNewton(woods,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('DogLeg')
x = {'p': np.tile([-3.0,-1.0],5000)}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = DogLeg(woods,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('cgTrust')
x = {'p': np.tile([-3.0,-1.0],5000)}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = cgTrust(woods,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('indef:')
print('Newton')
x = {'p': 2.0*np.ones((10000,))}
x['p'][0] = 1.0  
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'method': 'sppert'}
#[inform,path] = Newton(indef,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('LBFGS')
x = {'p': 2.0*np.ones((500,))}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-5,'m': 17}
[inform,path] = LBFGS(indef,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('TNewton')
x = {'p': 2.0*np.ones((500,))}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-5}
[inform,path] = TNewton(indef,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('DogLeg')
x = {'p': 2.0*np.ones((500,))}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 10}
[inform,path] = DogLeg(indef,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('cgTrust')
x = {'p': 2.0*np.ones((500,))}
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 10}
[inform,path] = cgTrust(indef,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('cragglvy:')
print('Newton')
x = {'p': 2.0*np.ones((10000,))}
x['p'][0] = 1.0  
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'method': 'sppert'}
#[inform,path] = Newton(cragglvy,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('LBFGS')
x = {'p': 2.0*np.ones((10000,))}
x['p'][0] = 1.0  
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'m': 17}
[inform,path] = LBFGS(cragglvy,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('TNewton')
x = {'p': 2.0*np.ones((10000,))}
x['p'][0] = 1.0  
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4}
[inform,path] = TNewton(cragglvy,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('DogLeg')
x = {'p': 2.0*np.ones((10000,))}
x['p'][0] = 1.0 
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = DogLeg(cragglvy,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

print('cgTrust')
x = {'p': 2.0*np.ones((10000,))}
x['p'][0] = 1.0 
globals.initialize()
starttime = timeit.default_timer()
nparams = {'maxit': 1000,'toler': 1.0e-4,'delbar': 100,'eta': 0.1,'initdel': 1}
[inform,path] = cgTrust(cragglvy,x,nparams)
probres(inform,path,sdparams)
print("The time difference is :", timeit.default_timer() - starttime)

