#-*- coding: utf8 -*-
import numpy as np
import scipy.odr.odrpack as odrpack
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
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
    general_sets.graph_format = ['b.', 'r-','g-','y--']
    
    '''----- sets1: Volume ------------'''
    sets1 = general_sets.clone()
    sets1.x_label = r'$t_{gas\ thermometer}$ in $^{\circ}C$'
    sets1.y_label = r'$t_{mercury\ thermomether}$ in $^{\circ}C$'
    
    '''----- sets2: Concentration -----'''
    sets2 = general_sets.clone()
    sets2.x_label = 'Sugar concentration (g/100ml)'
    sets2.y_label = 'Polarization axis rotation (deg)'

    return sets1, sets2

def linear(B,x):
    return B[0]*x + B[1]

def get_data():
    xls = pd.ExcelFile("../datasets/gasthermometer.xlsx")
    df1 = pd.read_excel(xls, sheet_name = 'Heat up')
    df2 = pd.read_excel(xls, sheet_name = 'Cool down')
    #return df1.drop(df1.index[0]), df2.drop(df2.index[0])
    return df1, df2


def plot_uncorrected_fit(sets, df):
    t_th, t_th_err = np.array(df['t_th'])[:-1], np.array(df['t_th_err'])[:-1]
    t_gas, t_gas_err = np.array(df['t_gas'])[:-1], np.array(df['t_gas_err'])[:-1]
    
    #ODR fitting
    linear_model = odrpack.Model(linear)
    data = odrpack.RealData(conc, rot, sx=conc_err, sy=rot_err)
    myodr = odrpack.ODR(data,linear_model, beta0=[3.1])
    output = myodr.run()
    params_n, params_s = output.beta, output.sd_beta
    params = unp.uarray(params_n, params_s)
    #print (params)

    popt, cov = curve_fit(ft.linear1, conc, rot, sigma = rot_err, absolute_sigma = True)
    x_fit = np.linspace(min(conc)*0.6, max(conc)*1.1, 1000)
    y_fit = linear(params_n, x_fit)
    y_fit_up = linear(params_n + params_s, x_fit)
    y_fit_down = linear(params_n - params_s, x_fit)
    y_fit_c = ft.linear1(x_fit, *popt)
    y_fit_c_up = ft.linear1(x_fit, *(popt + np.sqrt(cov[0][0])))
    y_fit_c_down = ft.linear1(x_fit, *(popt - np.sqrt(cov[0][0])))
    
    fmtr = ShorthandFormatter()
    label_str = fmtr.format('{0:.1u}', params[0])
    sets.graph_label = ['',r"$y=Ax,\quad with\ A=$" + label_str + " deg 100mL/g",'','']
    x_axis = [conc, x_fit]
    y_axis = [rot, y_fit]
    x_axis_err = [conc_err,[]]
    y_axis_err = [rot_err,[]]
    fig, ax = plot_multi_2(sets, x_axis, y_axis, x_axis_err, y_axis_err)
    ax.fill_between(x_fit,y_fit_up,y_fit_down, facecolor = "r", alpha = 0.1)
    #ax.fill_between(x_fit,y_fit_c_up,y_fit_c_down, facecolor = "r", alpha = 0.1)

    #ax.set_xbound(90,420)
    ax.set_xbound(9,42)
    ax.set_ybound(9,55)
    plt.subplots_adjust(left=0.12, right = 0.99, bottom = 0.14, top = 0.99)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)

def plot_both(sets, df1, df2):
    t_th_un = zip(list((df1['t_th'])[:-1]) + list((df2['t_th'])[:-1]), list((df1['t_th_err'])[:-1]) + list((df2['t_th_err'])[:-1]))
    t_gas_un = zip(list((df1['t_gas'])[:-1]) + list((df2['t_gas'])[:-1]), list((df1['t_gas_err'])[:-1]) + list((df2['t_gas_err'])[:-1]))
    t_th_un.sort(), t_gas_un.sort()

    t_th, t_th_err = zip(*t_th_un)[0], zip(*t_th_un)[1]
    t_gas, t_gas_err = zip(*t_gas_un)[0], zip(*t_gas_un)[1]

    fig, ax = plot_multi_2(sets, [t_gas,t_gas], [t_th,t_th], [t_gas_err,[]], [t_th_err,[]])
    ax.plot([0,100],[0,100], 'y--')

def plot_heat_up(sets, df):
    t_th, t_th_err = np.array(df['t_th'])[:-1], np.array(df2['t_th_err'])[:-1]
    t_gas, t_gas_err = np.array(df['t_gas'])[:-1], np.array(df2['t_gas_err'])[:-1]
    
    fig, ax = plot_multi_2(sets, [t_gas,t_gas], [t_th,t_th], [t_gas_err,[]], [t_th_err,[]])
    #ax.plot([0,100],[0,100], 'y--')

if __name__ == "__main__":
    sets1, sets2 = get_settings()
    df1, df2 = get_data()
    #plot_uncorrected(sets1, [df1,df2])
    plot_heat_up(sets1, df1)
    plt.show()