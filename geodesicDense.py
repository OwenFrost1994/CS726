import globals
import numpy as np
import scipy.sparse as sp

def geodesic(v,mode):
    z = ()
    m = len(v)
    if m & 1: 
        raise ValueError('must have even number of variables')
    else:
        n = m/2
    delt = 1/(n+1)
    x = np.concatenate(([globals.geodata['a']], v[0:m+1:2], [globals.geodata['c']]))
    y = np.concatenate(([globals.geodata['b']], v[1:m+1:2], [globals.geodata['d']]))

    if globals.geodata['rho'] == 1:
        if mode & 1:
            globals.numf += 1
            z += (np.sum(np.diff(x)**2+np.diff(y)**2)/delt,)
      
        if mode & 2:
            globals.numg += 1
            g = np.zeros((m,),dtype=float)
            g[0:m:2] = np.diff(x[:-1])-np.diff(x[1:])
            g[1:m:2] = np.diff(y[:-1])-np.diff(y[1:])
            z += ((2/delt)*g,)
      
        if mode & 4:
            globals.numH += 1
            z += ((2/delt)*sp.diags([-1., 2., -1], [-2, 0, 2], shape=(m, m)).toarray(),)
            # z += ((2/delt)*sp.diags([-1., 2.,-1.], [-2, 0, 2], shape=(m, m)),)
      
    else:
        rho = 1 + globals.geodata['alpha']*np.exp(-globals.geodata['beta']*(x[:-1]**2 + y[:-1]**2))
        if mode & 1:
            globals.numf += 1
            z += (np.sum(rho*(np.diff(x)**2+np.diff(y)**2))/delt,)
      
        if mode & 2:
            globals.numg += 1
            g = np.zeros((m,),dtype=float)
            for i in range(0,int(n)):
                j=i+1
                xju = x[j] - x[j+1]
                yju = y[j] - y[j+1]
                x2y2 = xju**2+yju**2
                g[2*i] = (x[j]-x[j-1])*rho[j-1] + xju*rho[j] - globals.geodata['beta']*x[j]*x2y2*(rho[j] - 1)
                g[2*i+1] = (y[j]-y[j-1])*rho[j-1] + yju*rho[j] - globals.geodata['beta']*y[j]*x2y2*(rho[j] - 1)
 
            z += ((2/delt)*g,)
      
        if mode & 4:
            globals.numH += 1
            H = np.zeros((m,m),dtype=float)
            for i in range(0,int(n)):
                j=i+1
                xju = x[j] - x[j+1]
                yju = y[j] - y[j+1]
                x2y2 = xju**2+yju**2
                ix = 2*i
                iy = 2*i+1
 
                betarhop = globals.geodata['beta']*(rho[j] - 1)
                H[ix,ix] = rho[j-1] + rho[j] + betarhop*(-4*xju*x[j]-x2y2+2*globals.geodata['beta']*x2y2*x[j]**2)
                H[ix,iy] = 2*betarhop*(-xju*y[j]-x[j]*yju+globals.geodata['beta']*x2y2*x[j]*y[j])
                H[iy,iy] = rho[j-1] + rho[j] + betarhop*(-4*yju*y[j]-x2y2+2*globals.geodata['beta']*x2y2*y[j]**2)
                H[iy,ix] = H[ix,iy]
                if (i < n-1):
                    H[ix+2,ix] = -rho[j]+2*betarhop*xju*x[j]
                    H[iy+2,iy] = -rho[j]+2*betarhop*yju*y[j]
                    H[ix,iy+2] = 2*betarhop*yju*x[j]
                    H[ix+2,iy] = 2*betarhop*xju*y[j]
                    H[ix,ix+2] = H[ix+2,ix]
                    H[iy,iy+2] = H[iy+2,iy]
                    H[iy+2,ix] = H[ix,iy+2]
                    H[iy,ix+2] = H[ix+2,iy]
            z += ((2/delt)*H,)
     
    return z

