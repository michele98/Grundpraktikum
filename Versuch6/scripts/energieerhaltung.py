#-*- coding: utf8 -*-
import numpy as np
from scipy.optimize import curve_fit
import scipy.odr.odrpack as odrpack
import matplotlib.pyplot as plt
from os import listdir
import math
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.umath import *
import uncertainties as uc

import sys
sys.path.insert(0,"../../scripts")
import data_manager as dm
import fit_functions as ft
from myPlot import Settings, plot, plot_multi_2, ShorthandFormatter


def get_settings():
    general_sets = Settings()
    
    general_sets.dataset_folder = "../datasets/"
    general_sets.dataset_file_name = "energieerhaltung.csv"
    general_sets.axes_label_fontsize = 20

    constant_sets = general_sets.clone()
    constant_sets.x_label = r"$\Delta\, x \:(cm)$"
    constant_sets.y_label = r"$a \:(cm\,s^{-2})$"

    constant_sets.graph_format = ["bs","r-","y--","y--"]
    #general_sets.fitted_graph_label = "Gefittete Kurve: " + r"$y = A\:\cos(\omega t + \phi)\: e^{-\beta t}$"
    constant_sets.graph_label = ["","","",""]

    line_sets = general_sets.clone()
    line_sets.x_label = r"$\Delta\, x \:(cm)$"
    line_sets.y_label = r"$v_1^2-v_2^2$"

    line_sets.graph_format = ["bs","r-","y--","y--"]
    #general_sets.fitted_graph_label = "Gefittete Kurve: " + r"$y = A\:\cos(\omega t + \phi)\: e^{-\beta t}$"
    line_sets.graph_label = ["","","",""]

    return constant_sets, line_sets


def get_data(sets):
    #reads data from datasets specified in Settings
    dataset = dm.csv_to_list(sets.dataset_folder + sets.dataset_file_name)
    
    x1_vals, x2_vals = dm.return_column(dataset, title = "x1"), dm.return_column(dataset, title = "x2")
    x_err = dm.return_column(dataset, title = "x_err")
    x1_u, x2_u = unp.uarray(x1_vals, x_err), unp.uarray(x2_vals, x_err)

    x0_vals = dm.return_column(dataset, title = "x0")
    x0_u = unp.uarray(x0_vals, 0.1)

    v1_vals, v2_vals = dm.return_column(dataset, title = "v1_corr"), dm.return_column(dataset, title = "v2_corr")
    v1_err, v2_err = dm.return_column(dataset, title = "v1_err"), dm.return_column(dataset, title = "v2_err")
    v1_u, v2_u = unp.uarray(v1_vals, v1_err), unp.uarray(v2_vals, v2_err)

    #return x0_u/100, x1_u/100, x2_u/100, v1_u/100, v2_u/100
    return x0_u, x1_u, x2_u, v1_u, v2_u


def plot_constant(sets):
    f = ft.constant
    fit_samples = 2
    x0,x1,x2,v1,v2 = get_data(sets)
    dv = v2-v1
    dx = x1-x0
    a = (v1**2-v2**2)/(2*(x1-x2))
    dx_n, x0_n, dv_n, a_n = unp.nominal_values(dx), unp.nominal_values(x0), unp.nominal_values(dv), unp.nominal_values(a)
    dx_s, x0_s, dv_s, a_s = unp.std_devs(dx), unp.std_devs(x0), unp.std_devs(dv), unp.std_devs(a)

    popt, cov = curve_fit(f, dv_n, a_n, sigma = a_s, absolute_sigma=True)
    a_err = np.sqrt(cov[0][0])
    x_fit_v = np.linspace(0, max(dv_n), fit_samples)
    #x_fit_x = np.linspace(min(x0_n), max(x0_n), fit_samples)
    x_fit_dx = np.linspace(0, max(dx_n)*1.1, fit_samples)
    y_fit = np.array([popt[0], popt[0]])

    y_fit_up, y_fit_down = y_fit+a_err, y_fit-a_err

    a_fit = ufloat(popt, a_err)
    fmtr = ShorthandFormatter()
    a_str = fmtr.format("{0:.1u}",a_fit)
    
    sets.graph_label[1] += "a = {} ".format(a_str) + r"$cm\,s^{-2}$"

    #fig, ax = plot_multi_2(sets, x_values = [dv_n,x_fit_v,x_fit_v,x_fit_v], y_values = [a_n,y_fit,y_fit_up,y_fit_down], x_err = [dv_s,[],[],[]], y_err = [a_s,[],[],[]])
    fig, ax = plot_multi_2(sets, x_values = [dx_n,x_fit_dx], y_values = [a_n,y_fit], x_err = [dx_s,[]], y_err = [a_s,[]])
    #fig, ax = plot_multi_2(sets, x_values = [dx_n,x_fit_dx,x_fit_dx, x_fit_dx], y_values = [a_n,y_fit,y_fit_up,y_fit_down], x_err = [x0_s,[],[],[]], y_err = [a_s,[],[],[]])

    ax.fill_between(x_fit_dx, y_fit_down, y_fit_up, facecolor = "r", alpha = 0.1)
    ax.set_xbound(0,70)
    b = ax.get_ybound()
    ax.set_ybound(b[0],b[1]+(b[1]-b[0])*0.1)
    plt.subplots_adjust(left=0.165, right = 0.98, bottom = 0.15, top = 0.99)
    
if __name__ == "__main__":
    sets1, sets2 = get_settings()
    plot_constant(sets1)

    plt.show()