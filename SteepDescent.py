import numpy as np
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
    
    for itera in range(params['maxit']):
        fi_k_t = fun(x['p'] + alfa * d, 1)[0]
        
        if fi_k_t > (x['f'] + params['ftol'] * alfa * x['g'].dot(d)):
            
            alphaR = alfa
            alfa = (alphaL + alphaR) / 2
            if abs(alphaL - alphaR) < params['xtol']:
                alpha_s = 1000
                status = 'ERROR'
                x_p = x['p'] + alpha_s * d
                x_n = {'p': x_p, 'f': fun(x_p, 1)[0], 'g': fun(x_p, 2)[0]}
                #print(alphaL, alphaR, abs(alphaL - alphaR), itera)
                return x_n, alpha_s, itera, status
            else:
                continue
        else:
            fi_k_t_d = fun(x['p'] + alfa * d, 2)[0].dot(d)
            if fi_k_t_d >= params['gtol'] * x['g'].dot(d):
                alpha_s = alfa
                status = 'OK'
                x_p = x['p'] + alpha_s * d
                x_n = {'p': x_p, 'f': fun(x_p, 1)[0], 'g': fun(x_p, 2)[0]}
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
        x_n = {'p': x_p, 'f': fun(x_p, 1)[0], 'g': fun(x_p, 2)[0]}
        return x_n, alpha_s, itera, status

import numpy as np
def SteepDescent(fun, x, sdparams):
    # sdparams = {'maxit': 1000,'toler': 1.0e-4}
    # The output inform is a dictionary containing two fields, status is 1 if the gradient tolerance
    # was achieved and 0 if not, iter is the number of steps taken and x is the solution structure,
    # with point, function and gradient evaluations at the solution point).
    Status = 0
    alpha = 1
    d = -x['g']
    params = {'ftol': 0.1,'gtol': 0.7,'xtol': 1e-6,'maxit': 100}
    
    for Itera in range(sdparams['maxit']):
        #print('Iternation-xk-dk-alpha:\n',Itera, x, d, alpha)
        x_n, alpha_s, itera, status = StepSize(fun, x, d, alpha, params)
        if status == 'ERROR':
            Status = 0
            #print(alpha_s, x_n)
            return x_n, Itera + 1, Status
        else:
            if np.linalg.norm(x['p']-x_n['p']) <= sdparams['toler']:
                Status = 1
                return x_n, Itera + 1, Status
            else:
                #print(alpha_s, x_n, x['g'].dot(d)*alpha/(x_n['g'].dot(-x_n['g'])))
                alpha = max(10 * params['xtol'], x['g'].dot(d)*alpha_s/(x_n['g'].dot(-x_n['g'])))
                x = x_n
                d = -x_n['g']
    
    if Itera == sdparams['maxit']-1:
        Status = 0
        return x_n, Itera + 1, Status