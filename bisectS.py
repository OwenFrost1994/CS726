import numpy as np
from numpy import array
import math

def bisect(f, a, b, tol):
    x1 = a
    x2 = b
    if f(array([x1]),2)[0][0]*f(array([x2]),2)[0][0] > 0:
        return 'Wrong Interval'
    else:
        nume = 0
        while abs(x1-x2) > tol:
            xm = (x1 + x2) / 2
            if f(array([x1]),2)[0]*f(array([xm]),2)[0] <0:
                x2 = xm;
            else:
                x1 = xm;
            nume += 1
        
        return x1, x2, nume