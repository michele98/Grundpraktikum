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
    general_sets.graph_format = ['b.', 'r-','g.','y-']
    general_sets.x_label = r'$t_{gas}$ in $^{\circ}C$'
    general_sets.y_label = r'$t_{Hg}$ in $^{\circ}C$'
    general_sets.title_fontsize = 18
    
    '''----- sets1: Eichkurve beim Aufwarmen ------------'''
    sets1 = general_sets.clone()
    #sets1.title = u"Eichurve beim Aufwärmen"
    
    '''----- sets2: Eichkurve beim Abkuhlen -----'''
    sets2 = general_sets.clone()
    #sets2.title = u"Eichkurve beim Abkühlen"

    '''----- sets3: Delta t beim Aufwarmen -----'''
    sets3 = general_sets.clone()
    #sets3.title = u"Differenz beim Aufwärmen"
    sets3.graph_format = ['b.', 'g.','r-']

    sets3.x_label = r'$t_{Hg}$ in $^{\circ}C$'
    sets3.y_label = r'$\Delta t = t_{gas} - t_{Hg}$ in K'
    
    '''----- sets3: Delta t beim Abkuhlen -----'''
    sets4 = sets3.clone()
    #sets4.title = u"Differenz beim Abkühlen"

    return sets1, sets2, sets3, sets4

def linear(B,x):
    return B[0]*x + B[1]

def get_data():
    xls = pd.ExcelFile("../datasets/gasthermometer.xlsx")
    df1 = pd.read_excel(xls, sheet_name = 'Heat up')
    df2 = pd.read_excel(xls, sheet_name = 'Cool down')
    #return df1.drop(df1.index[0]), df2.drop(df2.index[0])
    return df1, df2


def plot_calibration(sets, df):
    #list comprehension, because for some reason not all data is made of floats
    t_gas, t_gas_err = [float(a) for a in df['t_gas'][:-1]], [float(a) for a in df['t_gas_err'][:-1]]
    t_gas_c, t_gas_c_err = [float(a) for a in df['t_gas_corr'][:-1]], [float(a) for a in df['t_gas_corr_err'][:-1]]
    t_th, t_th_err = [float(a) for a in df['t_th'][:-1]], [float(a) for a in df['t_th_err'][:-1]]

    #ODR fitting
    linear_model = odrpack.Model(linear)
    data = odrpack.RealData(t_gas, t_th, sx=t_gas_err, sy=t_th_err)
    myodr = odrpack.ODR(data,linear_model, beta0=[1,1])
    output = myodr.run()
    params_n, params_s = output.beta, output.sd_beta
    params = unp.uarray(params_n, params_s)
    #fitting for corrected t_gas values
    data = odrpack.RealData(t_gas_c, t_th, sx=t_gas_c_err, sy=t_th_err)
    myodr = odrpack.ODR(data,linear_model, beta0=[1,1])
    output = myodr.run()
    params_c_n, params_c_s = output.beta, output.sd_beta
    params_c = unp.uarray(params_c_n, params_c_s)


    x_fit = np.linspace(min(t_gas), max(t_gas)*1.1, 10)
    y_fit = linear(params_n, x_fit)
    y_fit_up = linear(params_n + params_s, x_fit)
    y_fit_down = linear(params_n - params_s, x_fit)

    x_fit_c = np.linspace(min(t_gas_c)-10, max(t_gas_c)*1.1, 10)
    y_fit_c = linear(params_c_n, x_fit_c)
    y_fit_c_up = linear(params_c_n + params_c_s, x_fit_c)
    y_fit_c_down = linear(params_c_n - params_c_s, x_fit_c)
    #y_fit = ft.linear2(x_fit, *popt)
    
    fmtr = ShorthandFormatter()
    params_str = fmtr.format('{0:.1u}', params[0])
    params_str_b = fmtr.format('{0:.1u}', params[1])
    params_c_str = fmtr.format('{0:.1u}', params_c[0])
    params_c_str_b = fmtr.format('{0:.1u}', params_c[1])
    
    #sets.graph_label = ['',r"Aufw\"rmen, $y=Ax,\quad with\ A=$" + params_str, '', r"Abk\"hlen $y=Ax,\quad with\ A=$" + params_c_str]
    sets.graph_label = ['Unkorrigierte Werte',"$A = {}$, $b = {}\,K$".format(params_str, params_str_b), 'Korrigierte Werte', r"$A = {}, b = {}\,K$".format(params_c_str, params_c_str_b)]
    
    x_axis = [t_gas, x_fit, t_gas_c, x_fit_c]
    y_axis = [t_th, y_fit, t_th, y_fit_c]
    x_axis_err = [t_gas_err,[], t_gas_c_err, []]
    y_axis_err = [t_th_err,[], t_th_err, []]

    fig, ax = plot_multi_2(sets, x_axis, y_axis, x_axis_err, y_axis_err)
    ax.fill_between(x_fit,y_fit_up,y_fit_down, facecolor = "r", alpha = 0.2)
    ax.fill_between(x_fit_c,y_fit_c_up,y_fit_c_down, facecolor = "y", alpha = 0.4)

    ax.plot([0],[3], 'w.')
    legend = ax.legend(loc = sets.legend_location, fontsize = sets.legend_fontsize, title = 'Fit mit y = Ax + b')
    legend.get_title().set_fontsize('14')
    ax.set_xbound(-1,104)
    ax.set_ybound(-1,120)
    plt.subplots_adjust(left=0.15, right = 0.99, bottom = 0.16)

    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)

def plot_difference(sets, df):
    #list comprehension, because for some reason not all data is made of floats
    delta_t, delta_t_err = [float(a) for a in df['delta_t'][:-1]], [float(a) for a in df['delta_t_err'][:-1]]
    delta_t_c, delta_t_c_err = [float(a) for a in df['delta_t_corr'][:-1]], [float(a) for a in df['delta_t_corr_err'][:-1]]
    t_th, t_th_err = [float(a) for a in df['t_th'][:-1]], [float(a) for a in df['t_th_err'][:-1]]
    
    x_axis = [t_th, t_th, [-10,110]] #plots also a straight line at 0
    y_axis = [delta_t, delta_t_c, [0,0]]
    x_axis_err = [t_th_err, t_th_err,[]]
    y_axis_err = [delta_t_err, delta_t_c_err,[]]

    sets.graph_label = ['Unkorrigierte Werte','Korrigierte Werte','']

    fig, ax = plot_multi_2(sets, x_axis, y_axis, x_axis_err, y_axis_err)
    #ax.fill_between(x_fit,y_fit_up,y_fit_down, facecolor = "r", alpha = 0.2)
    #ax.fill_between(x_fit_c,y_fit_c_up,y_fit_c_down, facecolor = "y", alpha = 0.4)

    legend = ax.legend(loc = sets.legend_location, fontsize = sets.legend_fontsize, title = '')
    legend.get_title().set_fontsize('14')

    ax.set_xbound(-3,103)
    plt.subplots_adjust(left=0.15, right = 0.99, bottom = 0.15)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)

if __name__ == "__main__":
    sets1, sets2, sets3, sets4 = get_settings()
    df1, df2 = get_data() #df1 is heat up and df2 is cool down
    #plot_calibration(sets1, df1)
    plot_calibration(sets2, df2)
    #plot_difference(sets3, df1)
    plot_difference(sets4, df2)
    plt.show()