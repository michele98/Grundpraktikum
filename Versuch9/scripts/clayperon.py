#-*- coding: utf8 -*-
import numpy as np
import scipy.odr.odrpack as odrpack
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from os import listdir
import pandas as pd
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.umath import *
import uncertainties as uc

import sys
sys.path.insert(0,"../../scripts")
from myPlot import ShorthandFormatter
import fit_functions as ft
#def quadratic(x,a,b,c):
#    return (a*x+b)*x+c

def get_data():
    T = np.array([353, 363, 373, 383, 393])
    p = np.array([0.4736, 0.7010, 1.0133, 1.4327, 1.9853])*101.000
    vd = np.array([3.410, 2.360, 1.674, 1.211, 0.892])
    return T,p,vd

def compute_l(T, popt_p, popt_v):
    f = ft.cubic
    vf = 0.001
    l = T*dpdt(T,popt_p)*(f(T,*popt_v)-vf)
    return l

def dpdt(x, popt):
    return x*(x*3*popt[0] + 2*popt[1]) + popt[2]

def plot_pressure_curve():
    f = ft.cubic
    T, p, vd = get_data()
    fig, ax = plt.subplots()
    popt_p, cov_p = curve_fit(f,T,p)
    popt_v, cov_v = curve_fit(f,T,vd)
    ax.plot(T,p, 'b.')
    ax.plot(T,vd, 'g.')
    x_fit = np.linspace(min(T), max(T))
    ax.plot(x_fit, f(x_fit, *popt_p), 'r-')
    ax.plot(x_fit, f(x_fit, *popt_v), 'y-')
    
    #ax.plot(x_fit, dpdt(x_fit, popt_p))
    l = compute_l(371, popt_p, popt_v)
    print (l)

if __name__ == "__main__":
    plot_pressure_curve()
    plt.show()