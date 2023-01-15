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
    alphaR = 1000
    xf = fun(x['p'], 1)[0]
    xg = fun(x['p'], 2)[0]
    
    for itera in range(params['maxit']):
        fi_k_t = fun(x['p'] + alfa * d, 1)[0]
        
        if fi_k_t > (xf + params['ftol'] * alfa * xg.dot(d)):
            
            alphaR = alfa
            alfa = (alphaL + alphaR) / 2
            if abs(alphaL - alphaR) < params['xtol']:
                alpha_s = 1000
                status = 'ERROR'
                x_p = x['p'] + alpha_s * d
                x_n = {'p': x_p}
                #print(alphaL, alphaR, abs(alphaL - alphaR), itera)
                return x_n, alpha_s, itera, status
            else:
                continue
        else:
            fi_k_t_d = fun(x['p'] + alfa * d, 2)[0].dot(d)
            if fi_k_t_d >= params['gtol'] * xg.dot(d):
                alpha_s = alfa
                status = 'OK'
                x_p = x['p'] + alpha_s * d
                x_n = {'p': x_p}
                return x_n, alpha_s, itera, status
            else:
                alpha_L = alfa
            if alphaR > 500:
                alfa = 2 * alpha_L
                continue
    if itera == (params['maxit']-1):
        alpha_s = 1000
        status = 'ERROR'
        x_p = x['p'] + alpha_s * d
        x_n = {'p': x_p}
        return x_n, alpha_s, itera, status

def SteepDescent(fun, x, sdparams):
    # sdparams = {'maxit': 1000,'toler': 1.0e-4}
    # The output inform is a dictionary containing two fields, status is 1 if the gradient tolerance
    # was achieved and 0 if not, iter is the number of steps taken and x is the solution structure,
    # with point, function and gradient evaluations at the solution point).
    Status = 0
    alpha = 1
    params = {'ftol': 0.1,'gtol': 0.7,'xtol': 1e-6,'maxit': 100}
    
    for Itera in range(sdparams['maxit']):
        d = -fun(x['p'], 2)[0]
        #print('Iternation-xk-dk-alpha:\n',Itera, x, d, alpha)
        #print('Iternation-Norm of gradient:\n',Itera,np.linalg.norm(d))
        x_n, alpha_s, itera, status = StepSize(fun, x, d, alpha, params)
        if status == 'ERROR':
            Status = 0
            #print(alpha_s, x_n)
            return x_n, Itera + 1, Status
        else:
            if np.linalg.norm(fun(x_n['p'], 2)[0]) <= sdparams['toler']:#np.linalg.norm(x['p']-x_n['p']) <= sdparams['toler'] or 
                Status = 1
                return x_n, Itera + 1, Status
            else:
                xg = fun(x['p'], 2)[0]
                x_ng = fun(x_n['p'], 2)[0]
                #print(alpha_s, x_n, x['g'].dot(d)*alpha/(x_n['g'].dot(-x_n['g'])))
                alpha = max(10 * params['xtol'], xg.dot(d)*alpha_s/(x_ng.dot(-x_ng)))
                x = x_n
    
    if Itera == sdparams['maxit']-1:
        Status = 0
        return x_n, Itera + 1, Status
    
def Newton(fun, x, nparams):
    # sdparams = {'maxit': 1000,'toler': 1.0e-4, 'method': 'direct'}
    # The output inform is a dictionary containing two fields, status is 1 if the gradient tolerance
    # was achieved and 0 if not, iter is the number of steps taken and x is the solution structure,
    # with point, function and gradient evaluations at the solution point).
    Status = 0
    alpha = 1
    params = {'ftol': 1E-4,'gtol': 0.7,'xtol': 1e-6,'maxit': 100}
    
    for Itera in range(nparams['maxit']):
        #print(x['p'])
        funH = fun(x['p'], 4)[0] # Hessian
        fung = fun(x['p'], 2)[0] # Gradient
        #print(funH, fung)
        if nparams['method'] == 'direct':
            if type(funH).__name__ == 'csc_matrix':#csr_matrix
                d = -spsolve(funH,fung)
            else:
                d = -np.linalg.solve(funH,fung)
            numFact = 0
            while d.dot(fung) >= 0:
                funHn = funH + math.pow(2, numFact)*np.eye(funH.shape[0])
                numFact += 1
                D = np.abs(funHn.diagonal())
                D[D < 0.01] = 0.01
                D = np.diag(1./D)
                d = -D.dot(fung)
        if nparams['method'] == 'sppert':#pert
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
        x_n, alpha_s, itera, status = StepSize(fun, x, d, alpha, params)
        if status == 'ERROR':
            Status = 0
            return x_n, Itera + 1, Status
        else:
            if np.linalg.norm(fun(x_n['p'], 2)[0]) <= nparams['toler']:#np.linalg.norm(x['p']-x_n['p']) <=  or 
                Status = 1
                return x_n, Itera + 1, Status
            else:
                xg = fun(x['p'], 2)[0]
                x_ng = fun(x_n['p'], 2)[0]
                alpha = max(10 * params['xtol'], xg.dot(d)*alpha_s/(x_ng.dot(-x_ng)))
                x = x_n
        
    if Itera == nparams['maxit']-1:
        Status = 0
        return x_n, Itera + 1, Status

def BFGS(fun, x, sdparams):
    # sdparams = {'maxit': 1000,'toler': 1.0e-4}
    # The output inform is a dictionary containing two fields, status is 1 if the gradient tolerance
    # was achieved and 0 if not, iter is the number of steps taken and x is the solution structure,
    # with point, function and gradient evaluations at the solution point).
    Status = 0
    alpha = 1
    params = {'ftol': 0.1,'gtol': 0.7,'xtol': 1e-6,'maxit': 100}
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
        x_n, alpha_s, itera, status = StepSize(fun, x, d, alpha, params)
        if status == 'ERROR':
            Status = 0
            return x_n, Itera + 1, Status
        else:
            s = alpha_s * d
            x_n['p'] = x['p'] + s
            if np.linalg.norm(fun(x_n['p'], 2)[0]) <= sdparams['toler']:
                Status = 1
                return x_n, Itera + 1, Status
            else:
                f = fun(x_n['p'], 1)[0]
                fung_n = fun(x_n['p'], 2)[0]
                if  np.linalg.norm(fung_n, ord = np.inf)/min(1000, 1 + abs(f)) <= 1E-4:
                    Status = 1
                    return x_n, Itera + 1, Status
                else:
                    y = fung_n - fung
                    B_n = B + np.outer(y,y)/(y.dot(s)) - (B.dot(np.outer(s,s).dot(B)))/(s.dot(B.dot(s)))
                    alpha = max(10 * params['xtol'], fung.dot(d)*alpha_s/(fung_n.dot(-fung_n)))
                    B = B_n
                    x = x_n
        
    if Itera == sdparams['maxit']-1:
        Status = 0
        return x_n, Itera + 1, Status