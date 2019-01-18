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
    df = pd.read_excel(xls, sheet_name = 'Verdampfungswarme')
    time = np.array([float(a) for a in df['time'][1:]])
    time_err = np.array([float(a)/60 for a in df['time_err'][1:]]) #convert time from sec to min
    temp = np.array([float(a) for a in df['temp'][1:]])
    temp_err = np.array([float(a) for a in df['temp_err'][1:]])
    return time, time_err, temp, temp_err


def plot_curve(sets, f):
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