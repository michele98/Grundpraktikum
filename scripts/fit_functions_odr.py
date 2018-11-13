from numpy import exp, log
import numpy as np
import math
#using numpy instead of math because fitting has issues with non numpy functions

#param_bounds = (-np.inf,np.inf)

def fit(function_to_fit,x_values,y_values):
    #print("PARAM BOUNDS")
    #print (param_bounds)
    #popt, pcov = curve_fit(function_to_fit,x_values, y_values, bounds=param_bounds)
    return curve_fit(function_to_fit,x_values, y_values)

def constant(B,x):
    return B[0]

def linear1(B,x):
    return B[0]*x

def linear2(B,x):
    return B[0]*x+B[1]

def exp1(B,x):
    return B[0]*exp(B[1]*x)

def exp2(B,x):
#exponential function with offset
    return B[0]*exp(B[1]*x)+B[2]

def exp3(B,x):
    return B[0]*(1-exp(B[1]*x))

def exp4(B,x):
    return B[1]*(1-exp(B[1]*x))+B[2]

def ln(B,x):
    return B[0]*log(B[1]*x)

def quadratic(B,x):
#normal quadratic function
    return x*(B[0]*x+B[1])+B[2]

def cubic(B,x):
#a*x**3+b*x**2+c*x+d
    return B[3]+x*(B[2]+x*(B[1]+x*B[0]))

def quartic(B,x):
    return B[4]+x*(B[3]+x*(B[2]+x*(B[1]+x*B[0])))

def sqrt1 (B,x):
    try:
        return [B[0]*math.sqrt(k*B[1]) for k in x]
    except:
        return (B[0]*math.sqrt(x*B[1]))

def sqrt2 (B,x):
    try:
        return [B[0]*math.sqrt(k*B[1])+B[2] for k in x]
    except:
        return (B[0]*math.sqrt(x*B[1])+B[2])

def inversex(B,x):
    return a/(x+b)

def sin (B,x):
    try:
        #fit wants a list
        return [B[0]*math.sin(k/B[1]+B[2]) for k in x]
    except:
        #plotting wants a scalar
        return B[0]*(math.sin(x/B[1]+B[2]))

def xpowerN(B,x):
    return B[1]*(x**B[0])

def xpowerN2(B,x):
    return B[1]*(x**B[0])+B[2]

def identical (l):
    arr = np.array(l)
    return arr