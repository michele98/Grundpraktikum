#-*- coding: utf8 -*-
import numpy as np
import scipy.odr.odrpack as odrpack
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from os import listdir
import pandas as pd
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.umath import *
import uncertainties as uc

import sys
sys.path.insert(0,"../../scripts")
import data_manager as dm
import fit_functions as ft
from myPlot import Settings, plot_multi_2, ShorthandFormatter

def get_settings():
    general_sets = Settings()
    general_sets.graph_format = ['g.', 'r-','b-','k-.'] #daten, erster fit, zweiter Fit
    general_sets.graph_label = ['daten', 'fit before', 'fit after','']
    general_sets.x_label = r'Zeit in min'
    general_sets.y_label = r'Temperatur in $^{\circ}C$'
    general_sets.title_fontsize = 18
    
    '''----- sets1: Wasserwert ------------'''
    sets1 = general_sets.clone()
    sets1.title = u"Wasserwert"
    
    '''----- sets2: Dampf -----'''
    sets2 = general_sets.clone()
    sets2.title = u"Dampf"

    '''----- sets3: Eis -----'''
    sets3 = general_sets.clone()
    sets3.title = u"Eis"

    return sets1, sets2, sets3

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
    tm = p2[1] #comes from the fit
    t1 = p1[1]

    w = (cw*m2*(t2-tm)-cw*m1*(tm-t1))/(tm-t1)
    m_eq = w/cw
    print (w, m_eq)
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
    model = odrpack.Model(f)
    data = odrpack.RealData(x, y, sx=xs, sy=ys)
    myodr = odrpack.ODR(data, model, beta0=beta_0)
    output = myodr.run()
    params_n, params_s = output.beta, output.sd_beta
    return unp.uarray(params_n, params_s)


def plot_curve(sets):
    time, time_err, temp, temp_err = get_data()
    
    last_values = 15
    i = find_jump(temp)
    t_transition = (time[i:][0] + time[:i][-1])/2
    
    #normalizing transition at t = 0
    time -= t_transition
    t_transition = 0
    #a_before = a[:i]
    #a_after = a[i:]
    
    params_before = fit_odr(linear, time[:i], temp[:i], time_err[:i], temp_err[:i], [1,0])
    params_after = fit_odr(linear, time[-last_values:], temp[-last_values:], time_err[-last_values:], temp_err[-last_values:], [1,0])

    x_fit_before = np.linspace(min(time[:i]), max(time[:i]), 4)
    y_fit_before = linear(unp.nominal_values(params_before), x_fit_before)
    y_fit_before_up = linear(unp.nominal_values(params_before) + unp.std_devs(params_before), x_fit_before)
    y_fit_before_down = linear(unp.nominal_values(params_before) - unp.std_devs(params_before), x_fit_before)

    x_fit_after = np.linspace(min(time[i:]), max(time[i:]), 4)
    y_fit_after = linear(unp.nominal_values(params_after), x_fit_after)
    y_fit_after_up = linear(unp.nominal_values(params_after) + unp.std_devs(params_after), x_fit_after)
    y_fit_after_down = linear(unp.nominal_values(params_after) - unp.std_devs(params_after), x_fit_after)

    x_axis = [time, x_fit_before, x_fit_after, [t_transition, t_transition]]
    x_axis_err = [time_err,[],[],[]]
    y_axis = [temp, y_fit_before, y_fit_after, [0,30]]
    y_axis_err = [temp_err,[],[],[]]

    fig, ax = plot_multi_2(sets, x_axis, y_axis, x_axis_err, y_axis_err)
    #ax.fill_between(x_fit_before, y_fit_before_down, y_fit_before_up, facecolor = 'r', alpha = 0.2)
    #ax.set_xbound(-1,25.2)
    ax.set_ybound(8,28)
    w, m_eq = calculate_w(params_before, params_after, t_transition)

    fmtr = ShorthandFormatter()
    w_str = fmtr.format('{0:.2u}', w)
    m_eq_str = fmtr.format('{0:.2u}', m_eq)

    legend = ax.legend(loc = sets.legend_location, fontsize = sets.legend_fontsize, title = r'$m_{eq} = $' + m_eq_str + ' g')
    legend.get_title().set_fontsize('14')

    
if __name__ == "__main__":
    sets1, sets2, sets3 = get_settings()
    plot_curve(sets1)
    plt.show()