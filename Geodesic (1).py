import numpy as np
import globals
import math
class geodesic():
    def __init__(self, a = 1, b = 1, x0 = (-1,-1), x1 = (1,1), n = 5):
        self.alpha = a
        self.beta = b
        self.X0 = x0
        self.X1 = x1
        self.N = n
        
    def __call__ (self, Z, mode):
        z = ()
        globals.initialize()
        if mode & 1:
            globals.numf += 1    
            z = (self.value(Z),)
            
        if mode & 2:
            globals.numg += 1
            z = (self.gradient(Z),)
            
        #if mode & 4:
        #    globals.numH += 1
        return z
    
    def rho(self, x, y):
        value = 1 + self.alpha * math.exp(-self.beta*(x**2+y**2))
        return value
    def theta(self, x, y):
        value = self.alpha * math.exp(-self.beta*(x**2+y**2))
        return value
    
    def value(self, Z):
        deltat = 1/(self.N + 1)
        F = 0
        Xi = self.X0[0]
        Yi = self.X0[1]
        Xi1 = Z[0]
        Yi1 = Z[1]
        F +=  deltat * self.rho(Xi, Yi) * (((Xi1 - Xi)/deltat)**2 + ((Yi1 - Yi)/deltat)**2)
        for i in range(self.N):
            if i == self.N -1:
                Xi = Z[2*i]
                Yi = Z[2*i + 1]
                Xi1 = self.X1[0]
                Yi1 = self.X1[1]
            else:
                Xi = Z[2*i]
                Yi = Z[2*i + 1]
                Xi1 = Z[2*(i+1)]
                Yi1 = Z[2*(i+1) + 1]
            F += deltat * self.rho(Xi, Yi) * (((Xi1 - Xi)/deltat)**2 + ((Yi1 - Yi)/deltat)**2)
        return F
    
    def gradient(self, Z):
        deltat = 1/(self.N + 1)
        deltaF = []
        
        for i in range(self.N):
            if i == 0:
                Xi_1 = self.X0[0]
                Yi_1 = self.X0[1]
                Xi = Z[2*i]
                Yi = Z[2*i + 1]
                Xi1 = Z[2*(i+1)]
                Yi1 = Z[2*(i+1) + 1]
            else:
                if i == self.N-1:
                    Xi_1 = Z[2*(i-1)]
                    Yi_1 = Z[2*(i-1) + 1]
                    Xi = Z[2*i]
                    Yi = Z[2*i + 1]
                    Xi1 = self.X1[0]
                    Yi1 = self.X1[1]
                else:
                    Xi_1 = Z[2*(i-1)]
                    Yi_1 = Z[2*(i-1) + 1]
                    Xi = Z[2*i]
                    Yi = Z[2*i + 1]
                    Xi1 = Z[2*(i+1)]
                    Yi1 = Z[2*(i+1) + 1]
                
            parF_Xi = -2*self.beta*Xi*self.theta(Xi, Yi)*((Xi1 - Xi)**2/deltat + (Yi1 - Yi)**2/deltat) \
            + 2*self.rho(Xi,Yi)*(Xi - Xi1)/deltat + 2*self.rho(Xi_1,Yi_1)*(Xi - Xi_1)/deltat
            parF_Yi = -2*self.beta*Yi*self.theta(Xi, Yi)*((Xi1 - Xi)**2/deltat + (Yi1 - Yi)**2/deltat) \
            + 2*self.rho(Xi,Yi)*(Yi - Yi1)/deltat + 2*self.rho(Xi_1,Yi_1)*(Yi - Yi_1)/deltat
            deltaF.append(parF_Xi)
            deltaF.append(parF_Yi)
            
        deltaF_np = np.array(deltaF)
        return deltaF_np

import numpy as np
class geodesic_gradient_test():
    def __init__(self, a, b, x0, x1, n, h):
        self.alpha = a
        self.beta = b
        self.X0 = x0
        self.X1 = x1
        self.N = n
        self.H = h
    
    def __call__ (self, Z):
        z = ()
        vk = []
        for i in range(2*N):
            ek = np.zeros(2*N)
            ek[i] = 1
            vk.append((self.value(Z + self.H*ek)-self.value(Z))/self.H)
        return np.array(vk)
    
    def rho(self, x, y):
        value = 1 + self.alpha * math.exp(-self.beta*(x**2+y**2))
        return value
    def theta(self, x, y):
        value = self.alpha * math.exp(-self.beta*(x**2+y**2))
        return value
    
    def value(self, Z):
        deltat = 1/(self.N + 1)
        F = 0
        Xi = self.X0[0]
        Yi = self.X0[1]
        Xi1 = Z[0]
        Yi1 = Z[1]
        F +=  deltat * self.rho(Xi, Yi) * (((Xi1 - Xi)/deltat)**2 + ((Yi1 - Yi)/deltat)**2)
        for i in range(self.N):
            if i == self.N -1:
                Xi = Z[2*i]
                Yi = Z[2*i + 1]
                Xi1 = self.X1[0]
                Yi1 = self.X1[1]
            else:
                Xi = Z[2*i]
                Yi = Z[2*i + 1]
                Xi1 = Z[2*(i+1)]
                Yi1 = Z[2*(i+1) + 1]
            F += deltat * self.rho(Xi, Yi) * (((Xi1 - Xi)/deltat)**2 + ((Yi1 - Yi)/deltat)**2)
        return F