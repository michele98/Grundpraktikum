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
    general_sets.graph_format = ['b.', 'r-','b.','g-']
    
    '''----- sets1: Volume ------------'''
    sets1 = general_sets.clone()
    sets1.x_label = r'$\frac{\lambda}{2}$' + ' plate rotation (deg)'
    sets1.y_label = 'Polarization axis rotation (deg)'
    
    '''----- sets2: Concentration -----'''
    sets2 = general_sets.clone()
    sets2.x_label = r'$\frac{\lambda}{4}$' + ' plate rotation (deg)'
    sets2.y_label = 'Polarization axis rotation (deg)'

    return sets1, sets2

def linear(B,x):
    return B[0]*x

def linear2(B,x):
    return B[0]*x + B[1]

def get_data():
    xls = pd.ExcelFile("../datasets/lambda.xlsx")
    df1 = pd.read_excel(xls, sheet_name = 'Lambda2')
    df2 = pd.read_excel(xls, sheet_name = 'Lambda4')
    return df1, df2

def plot_lambda_2(sets, df):
    #p, p_err = np.array(df['plate'])[1:], np.array(df['plate_err'])[1:]
    #f, f_err = np.array(df['filter'])[1:], np.array(df['filter_err'])[1:]
    p, p_err = np.array(df['plate']), np.array(df['plate_err'])
    f, f_err = np.array(df['filter']), np.array(df['filter_err'])

    #ODR fitting
    linear_model = odrpack.Model(linear)
    data = odrpack.RealData(p, f, sx=p_err, sy=f_err)
    myodr = odrpack.ODR(data,linear_model, beta0=[2])
    output = myodr.run()
    params_n, params_s = output.beta, output.sd_beta
    params = unp.uarray(params_n, params_s)
    #print (params)

    x_fit = np.linspace(min(p), max(p)*1.1, 1000)
    y_fit = linear(params_n, x_fit)
    y_fit_up = linear(params_n + params_s, x_fit)
    y_fit_down = linear(params_n - params_s, x_fit)
    
    fmtr = ShorthandFormatter()
    label_str = fmtr.format('{0:.1u}', params[0])
    sets.graph_label = ['',r"$y=Ax,\quad with\ A=$" + label_str + " (deg L)/g",'','']
    x_axis = [p, x_fit]
    y_axis = [f, y_fit]
    x_axis_err = [p_err,[]]
    y_axis_err = [f_err,[]]
    fig, ax = plot_multi_2(sets, x_axis, y_axis, x_axis_err, y_axis_err)
    ax.fill_between(x_fit,y_fit_up,y_fit_down, facecolor = "r", alpha = 0.1)
    #ax.fill_between(x_fit,y_fit_c_up,y_fit_c_down, facecolor = "r", alpha = 0.1)

    ax.set_xbound(0,95)
    ax.set_ybound(0,190)
    plt.subplots_adjust(left=0.15, right = 0.99, bottom = 0.16, top = 0.99)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)

def plot_lambda_4(sets, df):
    p1, p1_err = np.array(df['plate'])[0:4], np.array(df['plate_err'])[0:4]
    f1, f1_err = np.array(df['filter'])[0:4], np.array(df['filter_err'])[0:4]
    p2, p2_err = np.array(df['plate'])[7:11], np.array(df['plate_err'])[7:11]
    f2, f2_err = np.array(df['filter'])[7:11], np.array(df['filter_err'])[7:11]

    #ODR fitting
    linear_model = odrpack.Model(linear)
    data = odrpack.RealData(p1, f1, sx=p1_err, sy=f1_err)
    myodr = odrpack.ODR(data,linear_model, beta0=[2])
    output = myodr.run()
    params1_n, params1_s = output.beta, output.sd_beta
    params1 = unp.uarray(params1_n, params1_s)
    #second data
    linear_model2 = odrpack.Model(linear2)
    data2 = odrpack.RealData(p2, f2, sx=p2_err, sy=f2_err)
    myodr2 = odrpack.ODR(data2,linear_model2, beta0=[-1,1])
    output2 = myodr2.run()
    params2_n, params2_s = output2.beta, output2.sd_beta
    params2 = unp.uarray(params2_n, params2_s)

    #popt, cov = curve_fit(ft.linear1, p, f, sigma = f_err, absolute_sigma = True)
    x_fit1 = np.linspace(min(p1), max(p1), 1000)
    y_fit1 = linear(params1_n, x_fit1)
    y_fit_up1 = linear(params1_n + params1_s, x_fit1)
    y_fit_down1 = linear(params1_n - params1_s, x_fit1)
    
    x_fit2 = np.linspace(min(p2), max(p2), 1000)
    y_fit2 = linear2(params2_n, x_fit2)
    y_fit_up2 = linear2(params2_n + params2_s, x_fit2)
    y_fit_down2 = linear2(params2_n - params2_s, x_fit2)
    
    fmtr = ShorthandFormatter()
    label1_str = fmtr.format('{0:.1u}', params1[0])
    label2_str1 = fmtr.format('{0:.1u}', params2[0])
    label2_str2 = fmtr.format('{0:.1u}', params2[1])
    sets.graph_label = ['',r"$y=Ax,\quad with\ A=$" + label1_str + " (deg L)/g", '', r"$y=Ax + b,\quad with\ A=$" + label2_str1 + " (deg L)/g\nand " + r'$b=$' + label2_str2]
    x_axis = [p1, x_fit1, p2, x_fit2]
    y_axis = [f1, y_fit1, f2, y_fit2]
    x_axis_err = [p1_err,[], p2_err, []]
    y_axis_err = [f1_err,[], f2_err, []]
    fig, ax = plot_multi_2(sets, x_axis, y_axis, x_axis_err, y_axis_err)
    ax.fill_between(x_fit1,y_fit_up1,y_fit_down1, facecolor = "r", alpha = 0.1)
    ax.fill_between(x_fit2,y_fit_up2,y_fit_down2, facecolor = "g", alpha = 0.1)
    #ax.fill_between(x_fit,y_fit_c_up,y_fit_c_down, facecolor = "r", alpha = 0.1)

    ax.set_xbound(-5,95)
    ax.set_ybound(-53,70)
    plt.subplots_adjust(left=0.15, right = 0.99, bottom = 0.16, top = 0.99)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
if __name__ == "__main__":
    sets1, sets2 = get_settings()
    df1, df2 = get_data()
    #plot_lambda_2(sets1, df1)
    plot_lambda_4(sets2, df2)
    plt.show()