#-*- coding: utf8 -*-
import numpy as np
import scipy.odr.odrpack as odrpack
#from scipy.optimize import curve_fit
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


def get_data():
    xls = pd.ExcelFile("../datasets/kalorimeter.xlsx")
    df = pd.read_excel(xls, sheet_name = 'Wasserwert')
    time = np.array([float(a) for a in df['time'][1:]])
    time_err = np.array([float(a)/60 for a in df['time_err'][1:]]) #convert time from sec to min
    temp = np.array([float(a) for a in df['temp'][1:]])
    temp_err = np.array([float(a) for a in df['temp_err'][1:]])
    return time, time_err, temp, temp_err

def calculate_w(p1, p2, t_transition):
    cw = 4.186 #Waermekpazitaet vom Wasser J/(g K)
    m1 = ufloat(252.18, 0.03) #in g
    m2 = ufloat(173.93, 0.03)
    t2 = ufloat(56,1) # in Celsius
    tm = linear(p2, t_transition) #comes from the fit
    t1 = linear(p1, t_transition)

    w = (cw*m2*(t2-tm)-cw*m1*(tm-t1))/(tm-t1)
    m_eq = w/cw
    #print (w, m_eq)
    return w, m_eq

def find_jump(y):
    jump, jump_max, index = 0,0,0
    for i in range(1,len(y)):
        jump = abs(y[i]-y[i-1])
        if jump > jump_max:
            index = i
            jump_max = jump
    return index

def linear(B, x):
    return B[0]*x + B[1]

def fit_odr(f,x,y, xs, ys, beta_0):
    #ODR fitting
    #print ('\n')
    model = odrpack.Model(f)
    data = odrpack.RealData(x, y, sx=xs, sy=ys)
    myodr = odrpack.ODR(data, model, beta0=beta_0)
    output = myodr.run()
    params_n, params_s = output.beta, output.sd_beta
    #output.pprint()
    res_var = output.res_var
    return unp.uarray(params_n, params_s), res_var


def plot_curve():
    time, time_err, temp, temp_err = get_data()
    fmtr = ShorthandFormatter()

    last_values = 20
    exclude_first = 2 #first 2 values to exclude from fit
    i = find_jump(temp)
    t_transition = (time[i-1] + time[i])/2
    
    #normalizing transition at t = 0
    #time -= t_transition
    #t_transition = 0
    #a_before = a[:i]
    #a_after = a[i:]
    
    params_before, res_before = fit_odr(linear, time[exclude_first:i], temp[exclude_first:i], time_err[exclude_first:i], temp_err[exclude_first:i], [1,0])
    params_before_str = [fmtr.format('{0:.1u}', a) for a in params_before]
    params_after, res_after = fit_odr(linear, time[-last_values:], temp[-last_values:], time_err[-last_values:], temp_err[-last_values:], [1,0])
    params_after_str = [fmtr.format('{0:.1u}', a) for a in params_after]

    x_fit_before = np.linspace(min(time[:i]), max(time[:i]), 4)
    y_fit_before = linear(unp.nominal_values(params_before), x_fit_before)
    y_fit_before_up = linear(unp.nominal_values(params_before) + unp.std_devs(params_before), x_fit_before)
    y_fit_before_down = linear(unp.nominal_values(params_before) - unp.std_devs(params_before), x_fit_before)

    x_fit_after = np.linspace(min(time[i:]), max(time[i:]), 4)
    y_fit_after = linear(unp.nominal_values(params_after), x_fit_after)
    y_fit_after_up = linear(unp.nominal_values(params_after) + unp.std_devs(params_after), x_fit_after)
    y_fit_after_down = linear(unp.nominal_values(params_after) - unp.std_devs(params_after), x_fit_after)

    #calculates Wasserwert
    w, m_eq = calculate_w(params_before, params_after, t_transition)
    w_str = fmtr.format('{0:.1u}', w/1000) #convert in kJ/K
    m_eq_str = fmtr.format('{0:.1u}', m_eq/1000) #convert in kg

    #calculates T1 and Tm
    t1 = linear(params_before, t_transition)
    tm = linear(params_after, t_transition)
    t1_str = fmtr.format('{0:.1u}', t1)
    tm_str = fmtr.format('{0:.1u}', tm)

    #setting up plot
    fig, ax = plt.subplots()
    ax.errorbar(time[exclude_first:i], temp[exclude_first:i], time_err[exclude_first:i], temp_err[exclude_first:i], 'b.', label = 'im Fit verwendete Datenpunkte')
    ax.errorbar(time[-last_values:], temp[-last_values:], time_err[-last_values:], temp_err[-last_values:], 'b.')
    ax.errorbar(time[:exclude_first], temp[:exclude_first], time_err[:exclude_first], temp_err[:exclude_first], 'y.', label = 'ausgelassene Datenpunkte im Fit')
    ax.errorbar(time[i:-last_values], temp[i:-last_values], time_err[i:-last_values], temp_err[i:-last_values], 'y.')

    ax.plot([],[], ' ', label = r'fit: $y=A\cdot x + b$') #nur um extra Text zu haben
    ax.plot(x_fit_before, y_fit_before, 'r-', label = r'$A={}\, ^\circ\! C/s$, $b={}\, ^\circ\! C$, $\chi^2={:1.3f}$'.format(params_before_str[0],params_before_str[1], res_before))
    ax.plot(time[i:], linear(unp.nominal_values(params_before), time[i:]), 'r--')
    ax.plot(x_fit_after, y_fit_after, 'g-', label = r'$A={}\, ^\circ\! C/s$, $b={}\, ^\circ\! C$, $\chi^2={:1.3f}$'.format(params_after_str[0],params_after_str[1], res_after))
    ax.plot(time[:i], linear(unp.nominal_values(params_after), time[:i]), 'g--')
    ax.plot([t_transition,t_transition], [0,30], 'k--', label = u'Übergangszeit: $T_1 = {}\, ^\circ\! C$, $T_m = {}\, ^\circ\! C$'.format(t1_str, tm_str))

    ax.fill_between(x_fit_before, y_fit_before_down, y_fit_before_up, facecolor = 'r', alpha = 0.2)
    ax.fill_between(x_fit_after, y_fit_after_down, y_fit_after_up, facecolor = 'g', alpha = 0.2)

    #ax.set_xbound(-13,11.8)
    ax.set_xbound(-0.5, 24.5)
    ax.set_ybound(8,28)
    ax.tick_params(labelsize = 18)
    
    #labels
    ax.set_xlabel('Zeit in min', fontsize = 18)
    ax.set_ylabel(r'Temperatur in $^{\circ}C$', fontsize = 18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend = ax.legend(bbox_to_anchor=(1.4, 0.5), loc='center right', borderaxespad=0., fontsize = 'large', title = r'$W = ${0} kJ/K, $m_{{eq}} = ${1} kg'.format(w_str, m_eq_str))
    legend.get_title().set_fontsize('14')

    fig.set_size_inches((10,5))
    plt.tight_layout()
    plt.subplots_adjust(left = 0.12, right = 0.74, bottom = 0.16, top = 0.99)

if __name__ == "__main__":
    plot_curve()
    plt.show()