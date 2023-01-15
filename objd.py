import globals
import math
import numpy as np

globals.initialize()

def objd(x,mode):
    z = ()
    r = np.linalg.norm(x)
    rm1 = r-1
    sqrm1 = rm1**2
    phi = math.atan(x[0]/x[1])
    
    if mode & 1:
        globals.numf += 1
        if (abs(rm1) < 1e-32):
            z += (0.,)
        else:
            z += (sqrm1 - 0.5*sqrm1*math.cos(1/rm1-phi),)
    
    if mode & 2:
        globals.numg += 1
        if (abs(rm1) < 1e-32):
            z += (np.array([0., 0.]),)
        else:
            sinrphi = math.sin(1/rm1-phi)
            dfdr = 2*rm1 - 0.5*(sinrphi + 2*rm1*math.cos(1/rm1-phi))
            dfdphi = -0.5*sqrm1*sinrphi
            rsq = r**2
            z += (np.array([x[0]*dfdr/r + dfdphi*x[1]/rsq, x[1]*dfdr/r - dfdphi*x[0]/rsq]),)
     
    if mode & 4:
        globals.numH += 1
        if (abs(rm1) < 1e-32):
            z += (np.array([[0., 0.], [0., 0.]]),)
        else:
            sinrphi = math.sin(1/rm1-phi)
            cosrphi = math.cos(1/rm1-phi)
            dfdr = 2*rm1 - 0.5*(sinrphi + 2*rm1*cosrphi)
            dfdphi = -0.5*sqrm1*sinrphi
            rsq = r**2
            drdx1 = x[0]/r
            drdx2 = x[1]/r
            dphidx1 = x[1]/rsq
            dphidx2 = -x[0]/rsq
            dargdx1 = -drdx1/sqrm1 - dphidx1
            dargdx2 = -drdx2/sqrm1 - dphidx2
            d2fdrdx1 = 2*drdx1 + (rm1*sinrphi - 0.5*cosrphi)*dargdx1 - drdx1*cosrphi
            d2fdphidx1 = -0.5*(2*rm1*sinrphi*drdx1 + sqrm1*cosrphi*dargdx1)
            d2fdrdx2 = 2*drdx2 + (rm1*sinrphi - 0.5*cosrphi)*dargdx2 - drdx2*cosrphi
            d2fdphidx2 = -0.5*(2*rm1*sinrphi*drdx2 + sqrm1*cosrphi*dargdx2)
            h11 = dfdr/r + x[0]*(r*d2fdrdx1 - dfdr*drdx1)/rsq + d2fdphidx1*x[1]/rsq - 2*dfdphi*drdx1*x[1]/r**3
            h12 = x[0]*(r*d2fdrdx2 - dfdr*drdx2)/rsq + (rsq*(d2fdphidx2*x[1]+dfdphi) - 2*r*drdx2*dfdphi*x[1])/rsq**2
            h21 = x[1]*(r*d2fdrdx1 - dfdr*drdx1)/rsq - (rsq*(d2fdphidx1*x[0]+dfdphi) - 2*r*drdx1*dfdphi*x[0])/rsq**2
            h22 = dfdr/r + x[1]*(r*d2fdrdx2 - dfdr*drdx2)/rsq - d2fdphidx2*x[0]/rsq + 2*dfdphi*drdx2*x[0]/r**3
            z += (np.array([[h11, h21], [h21, h22]]),)
      
    return z