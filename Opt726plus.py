import numpy as np
import math
import globals
import scipy
from scipy.sparse.linalg import spsolve
globals.initialize()

def StepSize(fun, x, d, alfa, params):
    # fun a pointer to a function (such as obja, objb, objc)
    # x a dictionary that on input has x['p'] set to the starting point values x = {'p': [-1.2, 1.0]}
    # and x['f'] and x['g'] set to the function and gradient values respectively.
    # d a vector containing the search direction
    # alfa the initial value of the step length
    # params a dictionary containing parameter values for the routine
    # params = {'ftol': 0.1,'gtol': 0.7,'xtol': 1e-6,: 100};
    #     Note that ftol and gtol were referred to as c1 and c2 in lectures. Return values are a
    #     dictionary inform that contains information about the run ('iter', 'status') and a new
    #     dictionary x containing the final solution values and function and gradient at that point.
    
    alphaL = 0
    alphaR = float('inf')
    xf = fun(x['p'], 1)[0]
    xg = fun(x['p'], 2)[0]
    inform = {'status': None, 'iter': -1}
    for itera in range(params['maxit']):
        fi_k_t = fun(x['p'] + alfa * d, 1)[0]
        
        if fi_k_t > (xf + params['ftol'] * alfa * np.dot(xg,d)):
            alphaR = alfa
            alfa = (alphaL + alphaR) / 2
            if abs(alphaL - alphaR) < params['xtol']:
                alpha_s = 1000
                status = 'ERROR'
                inform['status'] = 0
                inform['iter'] = itera
                inform['alpha_s'] = alpha_s
                x_p = x['p'] + alpha_s * d
                x_n = {'p': x_p}
                #print(alphaL, alphaR, abs(alphaL - alphaR), itera)
                return inform, x_n
            else:
                continue
        else:
            fi_k_t_d = np.dot(fun(x['p'] + alfa * d, 2)[0],d)
            if fi_k_t_d >= params['gtol'] * np.dot(xg,d):
                alpha_s = alfa
                status = 'OK'
                inform['status'] = 1
                inform['iter'] = itera
                inform['alpha_s'] = alpha_s
                x_p = x['p'] + alpha_s * d
                x_n = {'p': x_p}
                return inform, x_n
            else:
                alpha_L = alfa
            if alphaR > 500:
                alfa = 2 * alpha_L
                continue
    if itera == (params['maxit']-1):
        alpha_s = 1000
        inform['status'] = 0
        inform['iter'] = itera
        inform['alpha_s'] = alpha_s
        x_p = x['p'] + alpha_s * d
        x_n = {'p': x_p}
        return inform, x_n

def SteepDescent(fun, x, sdparams):
    # sdparams = {'maxit': 1000,'toler': 1.0e-4}
    # The output inform is a dictionary containing two fields, status is 1 if the gradient tolerance
    # was achieved and 0 if not, iter is the number of steps taken and x is the solution structure,
    # with point, function and gradient evaluations at the solution point).
    Status = 0
    alpha = 1
    params = {'ftol': 0.1,'gtol': 0.7,'xtol': 1e-6,'maxit': 100}
    Inform = {'status': None, 'iter': -1}
    for Itera in range(sdparams['maxit']):
        d = -fun(x['p'], 2)[0]
        #print('Iternation-xk-dk-alpha:\n',Itera, x, d, alpha)
        #print('Iternation-Norm of gradient:\n',Itera,np.linalg.norm(d))
        inform, x_n = StepSize(fun, x, d, alpha, params)
        status = 'OK' if inform['status'] == 1  else 'ERROR'
        alpha_s = inform['alpha_s']
        if status == 'ERROR':
            Status = 0
            #print(alpha_s, x_n)
            Inform['status'] = Status
            Inform['iter'] = Itera+1
            return Inform, x_n
        else:
            if np.linalg.norm(fun(x_n['p'], 2)[0]) <= sdparams['toler']:
                Status = 1
                Inform['status'] = Status
                Inform['iter'] = Itera+1
                return Inform, x_n
            else:
                xg = fun(x['p'], 2)[0]
                x_ng = fun(x_n['p'], 2)[0]
                #print(alpha_s, x_n, x['g'].dot(d)*alpha/(x_n['g'].dot(-x_n['g'])))
                alpha = max(10 * params['xtol'], np.dot(xg,d)*alpha_s/np.dot(x_ng,-x_ng))
                x = x_n
                d = -x_ng
    
    if Itera == sdparams['maxit']-1:
        Status = 0
        Inform['status'] = Status
        Inform['iter'] = Itera+1
        return Inform, x_n
    
def Newton(fun, x, nparams):
    # sdparams = {'maxit': 1000,'toler': 1.0e-4, 'method': 'direct'}
    # The output inform is a dictionary containing two fields, status is 1 if the gradient tolerance
    # was achieved and 0 if not, iter is the number of steps taken and x is the solution structure,
    # with point, function and gradient evaluations at the solution point).
    Status = 0
    alpha = 1
    params = {'ftol': 1E-4,'gtol': 0.7,'xtol': 1e-6,'maxit': 100}
    Inform = {'status': None, 'iter': -1}
    for Itera in range(nparams['maxit']):
        #print(x['p'])
        funH = fun(x['p'], 4)[0] # Hessian
        fung = fun(x['p'], 2)[0] # Gradient
        #print(funH, fung)
        if nparams['method'] == 'sppert':
            if type(funH).__name__ == 'ndarray':#csr_matrix
                d = -np.linalg.solve(funH,fung)
            else:
                d = -spsolve(funH,fung)
            numF = 0
            while d.dot(fung) >= 0:
                funHn = funH + math.pow(2, numF)*np.eye(fung.shape[0])
                numF += 1
                if type(funHn).__name__ == 'ndarray':#np.array
                    d = -np.linalg.solve(funHn,fung)
                else:
                    d = -spsolve(funHn,fung)
                globals.numFact += 1
        if nparams['method'] == 'direct':
            D = np.abs(funH.diagonal())
            D[D < 0.01] = 0.01
            D = np.diag(1./D)
            d = -D.dot(fung)
            #print(D, fung, d)
            numFact = 0
            while d.dot(fung) >= 0:
                funHn = funH + math.pow(2, numFact)*np.eye(funH.shape[0])
                numFact += 1
                D = np.abs(funHn.diagonal())
                D[D < 0.01] = 0.01
                D = np.diag(1./D)
                d = -D.dot(fung)
        #print('Iternation-xk-dk-alpha:\n',Itera, x, d, alpha)
        #print('Iternation-Norm of gradient:\n',Itera,np.linalg.norm(fung))
        inform, x_n = StepSize(fun, x, d, alpha, params)
        status = 'OK' if inform['status'] == 1  else 'ERROR'
        alpha_s = inform['alpha_s']
        if status == 'ERROR':
            Status = 0
            Inform['status'] = Status
            Inform['iter'] = Itera+1
            return Inform, x_n
        else:
            if np.linalg.norm(fun(x_n['p'], 2)[0]) <= nparams['toler']:#np.linalg.norm(x['p']-x_n['p']) <=  or 
                Status = 1
                Inform['status'] = Status
                Inform['iter'] = Itera+1
                return Inform, x_n
            else:
                xg = fun(x['p'], 2)[0]
                x_ng = fun(x_n['p'], 2)[0]
                alpha = max(10 * params['xtol'], xg.dot(d)*alpha_s/(x_ng.dot(-x_ng)))
                x = x_n
        
    if Itera == nparams['maxit']-1:
        Inform['status'] = Status
        Inform['iter'] = Itera+1
        return Inform, x_n

def BFGS(fun, x, sdparams):
    # sdparams = {'maxit': 1000,'toler': 1.0e-4}
    # The output inform is a dictionary containing two fields, status is 1 if the gradient tolerance
    # was achieved and 0 if not, iter is the number of steps taken and x is the solution structure,
    # with point, function and gradient evaluations at the solution point).
    Status = 0
    alpha = 1
    params = {'ftol': 0.1,'gtol': 0.7,'xtol': 1e-6,'maxit': 100}
    Inform = {'status': None, 'iter': -1}
    B = fun(x['p'], 4)[0] # Hessian
    if type(B).__name__ == 'csc_matrix':#csr_matrix
        B = B.toarray()
    x_n = x
    for Itera in range(sdparams['maxit']):
        fung = fun(x['p'], 2)[0] # Gradient
        funH = fun(x['p'], 4)[0] # Hessian
        if Itera == 0:
            s = alpha * fung
            x_n['p'] = x['p'] + s
            fung_n = fun(x_n['p'], 2)[0]
            y = fung_n - fung
            D = (s.dot(y)/(y.dot(y))) * np.eye(x['p'].shape[0])
            d = -D.dot(fung)
        else:
            d = -np.linalg.solve(B,fung)
        numF = 0
        while d.dot(fung) >= 0:
            funHn = funH + math.pow(2, numF)*np.eye(fung.shape[0])
            numF += 1
            if type(funHn).__name__ == 'ndarray':#np.array
                d = -np.linalg.solve(funHn,fung)
            else:
                d = -spsolve(funHn,fung)
            globals.numFact += 1
        #print('Iternation-xk-dk-alpha:\n',Itera, x, d, alpha)
        #print('Iternation-Norm of gradient:\n',Itera,np.linalg.norm(fung))
        alpha = 1
        inform, x_n = StepSize(fun, x, d, alpha, params)
        status = 'OK' if inform['status'] == 1  else 'ERROR'
        alpha_s = inform['alpha_s']
        if status == 'ERROR':
            Status = 0
            Inform['status'] = Status
            Inform['iter'] = Itera+1
            return Inform, x_n
        else:
            s = alpha_s * d
            x_n['p'] = x['p'] + s
            if np.linalg.norm(fun(x_n['p'], 2)[0]) <= sdparams['toler']:
                Status = 1
                Inform['status'] = Status
                Inform['iter'] = Itera+1
                return Inform, x_n
            else:
                f = fun(x_n['p'], 1)[0]
                fung_n = fun(x_n['p'], 2)[0]
                if  np.linalg.norm(fung_n, ord = np.inf)/min(1000, 1 + abs(f)) <= 1E-4:
                    Status = 1
                    Inform['status'] = Status
                    Inform['iter'] = Itera+1
                    return Inform, x_n
                else:
                    y = fung_n - fung
                    B_n = B + np.outer(y,y)/(y.dot(s)) - (B.dot(np.outer(s,s).dot(B)))/(s.dot(B.dot(s)))
                    #alpha = max(10 * params['xtol'], fung.dot(d)*alpha_s/(fung_n.dot(-fung_n)))
                    B = B_n
                    x = x_n
        
    if Itera == sdparams['maxit']-1:
        Status = 0
        Inform['status'] = Status
        Inform['iter'] = Itera+1
        return Inform, x_n
    
def LBFGS(fun, x, sdparams):
    Status = 0
    params = {'ftol': 0.1,'gtol': 0.7,'xtol': 1e-6,'maxit': 100}
    Inform = {'status': None, 'iter': -1}
    m = sdparams['m']
    Hkm0 = np.eye(x['p'].shape[0])
    sk = []
    yk = []
    rho = [0]*m
    eta = [0]*m
    for Itera in range(sdparams['maxit']):
        fung = fun(x['p'], 2)[0] # Gradient
        funH = fun(x['p'], 4)[0] # Hessian
        q = fung
        mov = max(0,Itera-m)
        for i in range(Itera-1,max(Itera-m-1,-1),-1):
            rhoi = 1/np.dot(yk[i-mov],sk[i-mov])
            rho[i-mov] = rhoi
            etai = rhoi*np.dot(sk[i-mov],q)
            eta[i-mov] = etai
            q = q - etai*yk[i-mov]
        r = np.dot(Hkm0,q)
        for i in range(max(Itera-m,0),Itera,1):
            rhoi = rho[i-mov]
            etai = eta[i-mov]
            beta = rhoi*np.dot(yk[i-mov],r)
            r = r + sk[i-mov]*(etai-beta)
        d = -r
        numF = 0
        while d.dot(fung) >= 0:
            funHn = funH + math.pow(2, numF)*np.eye(fung.shape[0])
            numF += 1
            if type(funHn).__name__ == 'ndarray':#np.array
                d = -np.linalg.solve(funHn,fung)
            else:
                d = -spsolve(funHn,fung)
            globals.numFact += 1
        
        alpha = 1
        inform, x_n = StepSize(fun, x, d, alpha, params)
        status = 'OK' if inform['status'] == 1  else 'ERROR'
        alpha_s = inform['alpha_s']
        #print(x['p'],d,alpha_s,rho)
        if status == 'ERROR':
            Status = 0
            Inform['status'] = Status
            Inform['iter'] = Itera+1
            return Inform, x_n
        else:
            s = alpha_s * d
            x_n['p'] = x['p'] + s
            if np.linalg.norm(fun(x_n['p'], 2)[0]) <= sdparams['toler']:
                Status = 1
                Inform['status'] = Status
                Inform['iter'] = Itera+1
                return Inform, x_n
            else:
                f = fun(x_n['p'], 1)[0]
                fung_n = fun(x_n['p'], 2)[0]
                if  np.linalg.norm(fung_n, ord = np.inf)/min(1000, 1 + abs(f)) <= 1E-4:
                    Status = 1
                    Inform['status'] = Status
                    Inform['iter'] = Itera+1
                    return Inform, x_n
                else:
                    y = fung_n - fung
                    if m<=Itera:
                        sk.pop(0)
                        yk.pop(0)
                    
                    sk.append(s)
                    yk.append(y)
                    Hkm0 = (np.dot(s,y)/np.dot(y,y))*np.eye(x['p'].shape[0])
                    #alpha = max(10 * params['xtol'], fung.dot(d)*alpha_s/(fung_n.dot(-fung_n)))
                    x = x_n
        
    if Itera == sdparams['maxit']-1:
        Status = 0
        Inform['status'] = Status
        Inform['iter'] = Itera+1
        return Inform, x_n
        
def TNewton(fun, x, sdparams):
    Status = 0
    params = {'ftol': 0.1,'gtol': 0.7,'xtol': 1e-6,'maxit': 100}
    Inform = {'status': None, 'iter': -1}
    
    for Itera in range(sdparams['maxit']):
        fung = fun(x['p'], 2)[0] # Gradient
        funH = fun(x['p'], 4)[0] # Hessian
        ei = min(0.5,math.sqrt(np.linalg.norm(fung)))*np.linalg.norm(fung)
        Bi = funH
        r0 = fung
        pi = TNewton_pi(Bi,r0,ei)
        
        alpha = 1
        inform, x_n = StepSize(fun, x, pi, alpha, params)
        status = 'OK' if inform['status'] == 1  else 'ERROR'
        alpha_s = inform['alpha_s']
        #print(x['p'],d,alpha_s,rho)
        if status == 'ERROR':
            Status = 0
            Inform['status'] = Status
            Inform['iter'] = Itera+1
            return Inform, x_n
        else:
            s = alpha_s * pi
            x_n['p'] = x['p'] + s
            if np.linalg.norm(fun(x_n['p'], 2)[0]) <= sdparams['toler']:
                Status = 1
                Inform['status'] = Status
                Inform['iter'] = Itera+1
                return Inform, x_n
            else:
                f = fun(x_n['p'], 1)[0]
                fung_n = fun(x_n['p'], 2)[0]
                
                y = fung_n - fung
                x = x_n
        
    if Itera == sdparams['maxit']-1:
        Status = 0
        Inform['status'] = Status
        Inform['iter'] = Itera+1
        return Inform, x_n
    
def TNewton_pi(Bi,r0,ei):
    sk = np.zeros(r0.shape)
    dk = -r0
    rk = r0
    k = 0
    while True:
        if np.dot(dk,Bi.dot(dk)) <= 0:
            if k == 0:
                pi = dk
                return pi
            else:
                pi = sk
                return pi
        alphak = np.dot(rk,rk)/np.dot(dk,Bi.dot(dk))
        sk1 = sk + alphak*dk
        rk1 = rk + alphak*Bi.dot(dk)
        if np.linalg.norm(rk1) <= ei:
            pi = sk1
            return pi
        betak = np.dot(rk1,rk1)/np.dot(rk,rk)
        dk1 = -rk1 + betak*dk
        
        sk = sk1
        rk = rk1
        dk = dk1
        
        k += 1
        globals.cgits += 1

def DogLeg(fun,x,sdparams):
    Status = 0
    Inform = {'status': None, 'iter': -1}
    delta = sdparams['initdel'] #delta_0
    delta_b = sdparams['delbar']
    eta = sdparams['eta']
    
    for Itera in range(sdparams['maxit']):
        f = fun(x['p'], 1)[0] # Function value f(xk)
        fung = fun(x['p'], 2)[0] # Gradient delta_f(xk)
        funH = fun(x['p'], 4)[0] # Hessian delta^2_f(xk)
        p = DogLeg_pk(fung,funH,delta,sdparams) # pk
        
        xn = x['p'] + p # xk+pk
        f_n = fun(xn, 1)[0] # f(xk+pk)
        rho = (f-f_n)/(f-(f+np.dot(fung,p)+0.5*np.dot(p,funH.dot(p)))) # rho = delta_fk/delta_mk
        
        if rho < 0.25:
            delta = delta/4
        else:
            if rho > 0.75 and abs(np.linalg.norm(p)- delta) < 1E-6:
                delta = min(2*delta,delta_b)
            else:
                delta = delta
        if rho > eta:
            x['p'] = xn
        else:
            x['p'] = x['p']
        if np.linalg.norm(fun(x['p'], 2)[0]) <= sdparams['toler']:
            Status = 1
            Inform['status'] = Status
            Inform['iter'] = Itera+1
            return Inform, x
        else:
            #y = fung_n - fung
            None
        
    if Itera == sdparams['maxit']-1:
        Status = 0
        Inform['status'] = Status
        Inform['iter'] = Itera+1
        return Inform, x
    
def DogLeg_pk(dxk,Bk,deltak,sdparams):
    if np.dot(dxk,Bk.dot(dxk)) <= 0 or np.linalg.norm(dxk)**3/np.dot(dxk,Bk.dot(dxk)) > deltak:
        pkc = -deltak*dxk/np.linalg.norm(dxk)
    else:
        pkc = -(np.linalg.norm(dxk)**2)*dxk/(np.dot(dxk,Bk.dot(dxk)))
    
    if type(Bk).__name__ == 'ndarray':#csr_matrix
        pkn = -np.linalg.solve(Bk,dxk)
    else:
        pkn = -spsolve(Bk,dxk)
    
    if pkn.dot(dxk) >= 0:
        if 'fail' in sdparams.keys() and sdparams['fail'] == 'cauchy':
            return pkc
        else:
            numF = 0
            while pkn.dot(dxk) >= 0:
                Bkn = Bk + math.pow(2, numF)*np.eye(dxk.shape[0])
                numF += 1
                if type(Bkn).__name__ == 'ndarray':#np.array
                    pkn = -np.linalg.solve(Bkn,dxk)
                else:
                    pkn = -spsolve(Bkn,dxk)
                globals.numFact += 1
    
    cita = (np.dot(pkc,(pkn-pkc)))**2 - (np.linalg.norm(pkn-pkc)**2)*(np.linalg.norm(pkc)**2 - deltak**2)
    r = min((-np.dot(pkc,(pkn-pkc)) + math.sqrt(cita))/(np.linalg.norm(pkn-pkc)**2),1)
    pks = pkc + r*(pkn -pkc)
    
    return pks

def cgTrust(fun,x,sdparams):
    Status = 0
    Inform = {'status': None, 'iter': -1}
    delta = sdparams['initdel'] #delta_0
    delta_b = sdparams['delbar']
    eta = sdparams['eta']
    
    for Itera in range(sdparams['maxit']):
        f = fun(x['p'], 1)[0] # Function value f(xk)
        fung = fun(x['p'], 2)[0] # Gradient delta_f(xk)
        funH = fun(x['p'], 4)[0] # Hessian delta^2_f(xk)
        #p = DogLeg_pk(fung,funH,delta,sdparams) # pk
        ei = min(0.5,math.sqrt(np.linalg.norm(fung)))*np.linalg.norm(fung)
        Bi = funH
        r0 = fung
        p = TNewton_pi(Bi,r0,ei)
        
        xn = x['p'] + p # xk+pk
        f_n = fun(xn, 1)[0] # f(xk+pk)
        rho = (f-f_n)/(f-(f+np.dot(fung,p)+0.5*np.dot(p,funH.dot(p)))) # rho = delta_fk/delta_mk
        
        if rho < 0.25:
            delta = delta/4
        else:
            if rho > 0.75 and abs(np.linalg.norm(p)- delta) < 1E-6:
                delta = min(2*delta,delta_b)
            else:
                delta = delta
        if rho > eta:
            x['p'] = xn
        else:
            x['p'] = x['p']
        if np.linalg.norm(fun(x['p'], 2)[0]) <= sdparams['toler']:
            Status = 1
            Inform['status'] = Status
            Inform['iter'] = Itera+1
            return Inform, x
        else:
            #y = fung_n - fung
            None
        
    if Itera == sdparams['maxit']-1:
        Status = 0
        Inform['status'] = Status
        Inform['iter'] = Itera+1
        return Inform, x