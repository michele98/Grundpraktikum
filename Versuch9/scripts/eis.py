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
    df1 = pd.read_excel(xls, sheet_name = 'Verdampfungswarme')
    df2 = pd.read_excel(xls, sheet_name = 'Schmelzwarme')
    dfs = {'Dampf': df1, 'Eis': df2}

    previous_datapoints = 8
    time_before = [float(a) for a in dfs['Dampf']['time'][-previous_datapoints:]]
    time_before_err = [float(a)/60 for a in dfs['Dampf']['time_err'][-previous_datapoints:]]
    temp_before = [float(a) for a in dfs['Dampf']['temp'][-previous_datapoints:]]
    temp_before_err = [float(a) for a in dfs['Dampf']['temp_err'][-previous_datapoints:]]

    time = [float(a)+time_before[-1] for a in dfs['Eis']['time'][1:]]
    time_err = [float(a)/60 for a in dfs['Eis']['time_err'][1:]] #convert time from sec to min
    temp = [float(a) for a in dfs['Eis']['temp'][1:]]
    temp_err = [float(a) for a in dfs['Eis']['temp_err'][1:]]

    time += time_before
    time_err += time_before_err
    temp += temp_before
    temp_err += temp_before_err

    return time, time_err, temp, temp_err


def eis(dfs):
    #das brauch ich, weil ich zusatzlich die Daten vom vorigen Versuch brauche
    previous_datapoints = 8
    time_before = [float(a) for a in dfs['Dampf']['time'][-previous_datapoints:]]
    time_before_err = [float(a)/60 for a in dfs['Dampf']['time_err'][-previous_datapoints:]]
    temp_before = [float(a) for a in dfs['Dampf']['temp'][-previous_datapoints:]]
    temp_before_err = [float(a) for a in dfs['Dampf']['temp_err'][-previous_datapoints:]]

    time = [float(a)+time_before[-1] for a in dfs['Eis']['time'][1:]]
    time_err = [float(a)/60 for a in dfs['Eis']['time_err'][1:]] #convert time from min to sec
    temp = [float(a) for a in dfs['Eis']['temp'][1:]]
    temp_err = [float(a) for a in dfs['Eis']['temp_err'][1:]]

    time += time_before
    time_err += time_before_err
    temp += temp_before
    temp_err += temp_before_err

    return time, time_err, temp, temp_err


def plot_wasserkurve(sets, f):
    time, time_err, temp, temp_err = f

    x_axis = [time]
    x_axis_err = [time_err]
    y_axis = [temp]
    y_axis_err = [temp_err]

    plot_multi_2(sets, x_axis, y_axis, x_axis_err, y_axis_err)

if __name__ == "__main__":
    sets1, sets2, sets3 = get_settings()
    #df1, df2, df3 = get_data()
    dfs = get_data()
    plot_wasserkurve(sets1, wasser(dfs))
    #plot_wasserkurve(sets2, dampf(dfs))
    #plot_wasserkurve(sets3, eis(dfs))
    plt.show()