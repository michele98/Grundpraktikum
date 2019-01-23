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
    p = np.array([0.4736, 0.7010, 1.0133, 1.4327, 1.9853])*100 #converting to kPa
    vd = np.array([3.410, 2.360, 1.674, 1.211, 0.892])
    return T,p,vd

def compute_l(T, popt_p, popt_v):
    vf = 0.001
    #f = ft.cubic
    #l = T*dpdt(T,popt_p)*(f(T,*popt_v)-vf)
    
    f = cubic
    l = T*dpdt(T,popt_p)*(f(popt_v,T)-vf)
    return l

def dpdt(x, popt):
    return x*(x*3*popt[0] + 2*popt[1]) + popt[2]

def fit_odr(f,x,y, beta_0):
    model = odrpack.Model(f)
    data = odrpack.RealData(x, y)
    myodr = odrpack.ODR(data, model, beta0=beta_0)
    output = myodr.run()
    output.pprint()
    return output

def cubic(B,x):
    #return B[0]*x*x*x + B[1]*x*x + B[2]*x + B[3]
    return ((B[0]*x+B[1])*x+B[2])*x+B[3]

def plot_pressure_curve():
    #f = ft.cubic
    f = cubic
    T_normal = 373.15
    T_labor = ufloat(371, 0.5)
    T, p, vd = get_data()
    #popt_p, cov_p = curve_fit(f,T,p)
    #popt_v, cov_v = curve_fit(f,T,vd)
    out_p = fit_odr(f,T,p, [1,1,1,1])
    out_v = fit_odr(f,T,vd, [1,1,1,1])
    popt_p, chi2_p = out_p.beta, out_p.res_var
    popt_v, chi2_v = out_v.beta, out_v.res_var

    l_normal = compute_l(T_normal, popt_p, popt_v)
    l_labor = compute_l(T_labor, popt_p, popt_v)
    fmtr = ShorthandFormatter()
    l_labor_str = fmtr.format('{0:.1u}', l_labor)
    #l_normal_str = 

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    #ax2.plot(T,vd, 'g.')

    ax.plot(T,p, 'bo', label = 'Werte der Dampfdruckkurve vom Skript')
    x_fit = np.linspace(min(T)-20, max(T)+20)
    #ax.plot(x_fit, f(x_fit, *popt_p), 'r-', label = 'p(T), gefittet mit $y=ax^3+bx^2+cx+d$: \n $a={:.4f},\ b={:.2f}$,\n $c={:.2f},\ d={:.2f}$'.format(popt_p[0], popt_p[1], popt_p[2], popt_p[3]))
    ax.plot(x_fit, f(popt_p, x_fit), 'r-', label = 'p(T), gefittet mit $y=ax^3+bx^2+cx+d$: \n $a={:.4f},\ b={:.2f}$,\n $c={:.2f},\ d={:.2f}$'.format(popt_p[0], popt_p[1], popt_p[2], popt_p[3]))
    #ax.plot(x_fit, f(x_fit, *popt_p), 'r-', label = 'p(T), gefittet mit $y=ax^2+bx+c$: \n $a={:.2f},\ b={:.2f}$,\n $c={:.2f}$'.format(popt_p[1], popt_p[2], popt_p[3]))
    #ax.plot(x_fit, f(x_fit, *popt_v), 'y-')
    ax2.plot(x_fit, dpdt(x_fit, popt_p), 'g-', label = r'$\frac{dp}{dt}$, erhalten durch Ableiten von p(T)')
    
    ax2.plot([], [],' ', label = '$l_{{Labor}} = {}\: J/(kg\,K)$ bei $T=371.0(5)\,K$, \n $l_{{normal}} = {:.1f}$ bei $T=373.15\,K$'.format(l_labor_str, l_normal))
    ax.set_xlabel('Temperatur in K', fontsize = 18)
    ax.set_ylabel('Druck in kPa', fontsize = 18)
    ax.set_xbound(351, 395)
    ax.set_ybound(40, 210)
    ax.tick_params(labelsize = 18)

    ax2.set_ylabel(r'$\frac{dp}{dt}\ in\ \frac{kPa}{K}$', fontsize = 18, rotation = 90)
    ax2.tick_params(labelsize = 18)
    ax2.yaxis.labelpad = -5

    legend = ax.legend(fontsize = 13)
    legend.get_title().set_fontsize('14')
    ax2.legend(fontsize = 13, loc = 'center left')
    plt.tight_layout()
    plt.subplots_adjust(left = 0.14, right = 0.88, bottom = 0.13, top = 0.99)


if __name__ == "__main__":
    plot_pressure_curve()
    plt.show()