from numpy import exp, log
import numpy as np
import math
from scipy.optimize import curve_fit
#using numpy instead of math because fitting has issues with non numpy functions

#param_bounds = (-np.inf,np.inf)

def fit(function_to_fit,x_values,y_values):
    #print("PARAM BOUNDS")
    #print (param_bounds)
    #popt, pcov = curve_fit(function_to_fit,x_values, y_values, bounds=param_bounds)
    return curve_fit(function_to_fit,x_values, y_values)

def constant(x,a):
    return a

def linear1(x,a):
    return a*x

def linear2(x,a,b):
    return a*x+b

def exp1(x,A,b):
    return A*exp(b*x)

def exp2(x,A,b,c):
#exponential function with offset
    return A*exp(b*x)+c

def exp3(x,A,b):
    return A*(1-exp(b*x))

def exp4(x,A,b,c):
    return A*(1-exp(b*x))+c

def ln(x,A,b):
    return A*log(b*x)

def quadratic(x,a,b,c):
#normal quadratic function
    return x*(a*x+b)+c

def cubic(x,a,b,c,d):
#a*x**3+b*x**2+c*x+d
    return d+x*(c+x*(b+x*a))

def quartic(x,a,b,c,d,e):
    return e+x*(d+x*(c+x*(b+x*a)))

def sqrt1 (x,a,b):
    try:
        return [a*math.sqrt(k*b) for k in x]
    except:
        return (a*math.sqrt(x*b))

def sqrt2 (x,a,b,c):
    try:
        return [a*math.sqrt(k*b)+c for k in x]
    except:
        return (a*math.sqrt(x*b)+c)

def inversex(x,a,b):
    return a/(x+b)

def sin (x, a, b, c):
    try:
        #fit wants a list
        return [a*math.sin(k/b+c) for k in x]
    except:
        #plotting wants a scalar
        return a*(math.sin(x/b+c))

def xtoN(x,a,B):
    return B*(x**a)

def xtoN2(x,a,B,c):
    return B*(x**a)+c

def identical (l):
    arr = np.array(l)
    return arr