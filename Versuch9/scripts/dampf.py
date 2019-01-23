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


def get_data():
    xls = pd.ExcelFile("../datasets/kalorimeter.xlsx")
    df = pd.read_excel(xls, sheet_name = 'Verdampfungswarme')
    time = np.array([float(a) for a in df['time'][1:]])
    time_err = np.array([float(a)/60 for a in df['time_err'][1:]]) #convert time from sec to min
    temp = np.array([float(a) for a in df['temp'][1:]])
    temp_err = np.array([float(a) for a in df['temp_err'][1:]])
    return time, time_err, temp, temp_err

def linear(B, x):
    return B[0]*x + B[1]

def calculate_l(p1, pm, t_transition):
    cw = 4.186 #Waermekpazitaet vom Wasser J/(g K)
    w = ufloat(131, 41) #J/K
    m1 = ufloat(331.17, 0.03) #in g
    m2 = ufloat(25.53, 0.03) #in g
    t2 = ufloat(98,0.2) # in Celsius
    t1 = linear(p1, t_transition)
    tm = linear(pm, t_transition) #comes from the fit
    print (t1,t2, tm)
    l = ((cw*m1 + w)*(tm-t1)-cw*m2*(t2-tm))/m2
    return l

def find_transition(x,y,p1,p2):
    #t_room = ufloat(22,0.5) #Zimmertemperatur in Celsius
    dim = len(x)
    #Al, Ar = 0,0
    min_index = 0
    m = 10000
    for j in range(1,dim):
        Ar, Al = 0,0
        for i in range(1,j):
            Al += (x[i]-x[i-1])*((y[i]+y[i-1])/2-linear(p1, (x[i]+x[i-1])/2))
        for i in range(1,dim-j):
            Ar -= (x[dim-i]-x[dim-i-1])*((y[dim-i]+y[dim-i-1])/2-linear(p2, (x[dim-i]+x[dim-i-1])/2))
        if abs(Al-Ar)<m:
            min_index = j
            m = abs(Al-Ar)
    return min_index

def find_jumps(y):
    threshold = 2
    i1_set = False
    i1, i2 = 0,0
    equals, equals2 = 0,0 #needed to avoid comparison of same values
    for i in range(2,len(y)-4):
        if y[i] == y[i-1]:
            equals += 1
        else:
            equals = 0
        if y[i-1-equals] == y[i-2-equals]:
            equals2 += 1
        else:
            equals2 = 0
        #print (equals, equals2)
        if y[i-equals-1]!=y[i-2-equals2]:
            jump_ratio = abs((y[i]-y[i-1-equals])/(y[i-equals-1]-y[i-2-equals2]))
            if jump_ratio > threshold:
                if not i1_set:
                    i1 = i
                    i1_set = True
            elif jump_ratio < 0.3:
                i2 = i
    print (i1, i2)
    return i1, i2

def fit_odr(f,x,y, xs, ys, beta_0):
    #ODR fitting
    #print ('\n')
    model = odrpack.Model(f)
    data = odrpack.RealData(x, y, sx=xs, sy=ys)
    myodr = odrpack.ODR(data, model, beta0=beta_0)
    output = myodr.run()
    output.pprint()
    return output

def plot_curve():
    time, time_err, temp, temp_err = get_data()
    fmtr = ShorthandFormatter()
    
    i, i2 = find_jumps(temp)
    #i = 10
    last_values = 15 #len(temp)-i2
    exclude_first = 1 #first 2 values to exclude from fit
    #t_transition = (time[i-1] + time[i])/2

    output_before = fit_odr(linear, time[exclude_first:i], temp[exclude_first:i], time_err[exclude_first:i], temp_err[exclude_first:i], [1,0])
    output_after = fit_odr(linear, time[-last_values:], temp[-last_values:], time_err[-last_values:], temp_err[-last_values:], [1,0])
    
    #params_before = fit_odr(linear, time[exclude_first:i], temp[exclude_first:i], time_err[exclude_first:i], temp_err[exclude_first:i], [1,0])
    #params_before_str = [fmtr.format('{0:.1u}', a) for a in params_before]
    #params_after = fit_odr(linear, time[-last_values:], temp[-last_values:], time_err[-last_values:], temp_err[-last_values:], [1,0])
    #params_after_str = [fmtr.format('{0:.1u}', a) for a in params_after]

    #verdampfungswarme ausrechnen
    i_transition = find_transition(time, temp, output_before.beta, output_after.beta)
    t_transition = (time[i_transition]+time[i_transition-1])/2
    #normalizing transition at t = 0
    #time -= t_transition
    #t_transition = 0

    #output_before = fit_odr(linear, time[exclude_first:i], temp[exclude_first:i], time_err[exclude_first:i], temp_err[exclude_first:i], [1,0])
    #output_after = fit_odr(linear, time[-last_values:], temp[-last_values:], time_err[-last_values:], temp_err[-last_values:], [1,0])

    params_before = unp.uarray(output_before.beta, output_before.sd_beta)
    params_before_str = [fmtr.format('{0:.1u}', a) for a in params_before]
    params_after = unp.uarray(output_after.beta, output_after.sd_beta)
    params_after_str = [fmtr.format('{0:.1u}', a) for a in params_after]
    res_before, res_after = output_before.res_var, output_after.res_var

    x_fit_before = np.linspace(min(time[:i+1]), max(time[:i+1]), 4)
    y_fit_before = linear(unp.nominal_values(params_before), x_fit_before)
    y_fit_before_up = linear(unp.nominal_values(params_before) + unp.std_devs(params_before), x_fit_before)
    y_fit_before_down = linear(unp.nominal_values(params_before) - unp.std_devs(params_before), x_fit_before)

    x_fit_after = np.linspace(min(time[-last_values-1:]), max(time[-last_values-1:]), 4)
    y_fit_after = linear(unp.nominal_values(params_after), x_fit_after)
    y_fit_after_up = linear(unp.nominal_values(params_after) + unp.std_devs(params_after), x_fit_after)
    y_fit_after_down = linear(unp.nominal_values(params_after) - unp.std_devs(params_after), x_fit_after)


    l = calculate_l(params_before, params_after, t_transition)
    l_str = fmtr.format('{0:.1u}', l/1000) #convert from kJ/kg to MJ/kg

    #setting up plot
    fig, ax = plt.subplots()
    ax.errorbar(time[exclude_first:i], temp[exclude_first:i], time_err[exclude_first:i], temp_err[exclude_first:i], 'b.', label = 'im Fit verwendete Datenpunkte')
    ax.errorbar(time[-last_values:], temp[-last_values:], time_err[-last_values:], temp_err[-last_values:], 'b.')
    ax.errorbar(time[:exclude_first], temp[:exclude_first], time_err[:exclude_first], temp_err[:exclude_first], 'b.')
    ax.errorbar(time[i:-last_values], temp[i:-last_values], time_err[i:-last_values], temp_err[i:-last_values], 'y.', label = 'ausgelassene Datenpunkte im Fit')

    ax.plot([],[], ' ', label = r'fit: $y=A\cdot x + b$') #nur um extra Text zu haben
    ax.plot(x_fit_before, y_fit_before, 'r-', label = r'$A={}\, ^\circ\! C/s$, $b={}\, ^\circ\! C$, $\chi^2={:1.3f}$'.format(params_before_str[0],params_before_str[1], res_before))
    ax.plot(time[i:], linear(unp.nominal_values(params_before), time[i:]), 'r--')
    ax.plot(x_fit_after, y_fit_after, 'g-', label = r'$A={}\, ^\circ\! C/s$, $b={}\, ^\circ\! C$, $\chi^2={:1.3f}$'.format(params_after_str[0],params_after_str[1], res_after))
    ax.plot(time[:-last_values], linear(unp.nominal_values(params_after), time[:-last_values]), 'g--')
    ax.plot([t_transition,t_transition], [10,50], 'k--', label = u'Übergangszeit' + r'$\:={:.1f}(3)$ min'.format(t_transition))

    ax.fill_between(x_fit_before, y_fit_before_down, y_fit_before_up, facecolor = 'r', alpha = 0.2)
    ax.fill_between(x_fit_after, y_fit_after_down, y_fit_after_up, facecolor = 'g', alpha = 0.2)

    ax.set_xbound(-0.5,25.5)
    ax.set_ybound(10,49)
    ax.tick_params(labelsize = 18)
    #labels
    ax.set_xlabel('Zeit in min', fontsize = 18)
    ax.set_ylabel(r'Temperatur in $^{\circ}C$', fontsize = 18)
    plt.tight_layout()
    plt.subplots_adjust(left = 0.12, right = 0.99, bottom = 0.16, top = 0.99)

    legend = ax.legend(loc = 'center left', fontsize = 'large', title = r'$l = ${0} MJ/kg'.format(l_str))
    legend.get_title().set_fontsize('14')


if __name__ == "__main__":
    plot_curve()
    plt.show()