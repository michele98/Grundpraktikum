#-*- coding: utf8 -*-
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from os import listdir
import pandas
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
    general_sets.dataset_file_name = "mittlere_geschwindigkeit.csv"

    general_sets.x_label = r"$D^2 \:(m^2)$"
    general_sets.y_label = u"T (s)"

    general_sets.graph_format = ["b.","r-","y--","y--"]
    #general_sets.fitted_graph_label = "Gefittete Kurve: " + r"$y = A\:\cos(\omega t + \phi)\: e^{-\beta t}$"
    general_sets.graph_label = ["", "Linear","",""]
    general_sets.axes_label_fontsize = 20

    return general_sets


def get_data(sets):
    #reads data from datasets specified in Settings
    dataset = dm.csv_to_list(sets.dataset_folder + sets.dataset_file_name)
    
    distances_vals = dm.return_column(dataset, title = "d")
    distances_err = dm.return_column(dataset, title = "d_err")
    distances_u = unp.uarray(distances_vals, distances_err)

    times_vals = dm.return_column(dataset, title = "t_avg")
    times_err = dm.return_column(dataset, title = "t_dev")
    times_u = unp.uarray(times_vals, times_err)

    return distances_u/100, times_u

def plot_line(sets):
    f = ft.linear1
    fit_samples = 10
    distances, times = get_data(sets)
    distances2 = distances**2
    distances2_n, times_n = unp.nominal_values(distances2), unp.nominal_values(times)
    distances2_s, times_s = unp.std_devs(distances2), unp.std_devs(times)

    popt, cov = curve_fit(f, times_n, distances2_n, sigma = times_s, absolute_sigma=True)
    #inv_params = unp.uarray([popt[0], popt[1]], [cov[0][0], cov[1][1]])
    #true_params = [1/inv_params[0], -inv_params[1]/inv_params[0]]
    inv_params = unp.uarray(popt, [np.sqrt(cov[i][i]) for i in range(len(cov))])
    true_params = 1/inv_params
    print(inv_params, true_params)

    x_fit = np.linspace(min(distances2_n), max(distances2_n), fit_samples)
    y_fit = f(x_fit, *unp.nominal_values(true_params))
    y_fit_up = f(x_fit, *(unp.nominal_values(true_params)+unp.std_devs(true_params)))
    y_fit_down = f(x_fit, *(unp.nominal_values(true_params)-unp.std_devs(true_params)))

    #y_fit_up, y_fit_down = y_fit + unp.std_devs(true_params), y_fit - unp.std_devs(true_params)

    #print (y_fit_up,y_fit_down, y_fit)
    fmtr = ShorthandFormatter()
    a_str = fmtr.format("{0:.1u}",true_params[0])

    sets.graph_label[1] = r"$a={}$".format(a_str)
    fig, ax = plot_multi_2(sets, x_values = [distances2_n,x_fit], y_values = [times_n,y_fit], x_err = [distances2_s,[]], y_err = [times_s,[]])


    x_fit = np.linspace(min(times_n), max(times_n), fit_samples)
    y_fit = f(x_fit, *popt)
    sets.x_label = "T (s)"
    sets.y_label = r"$D^2 \:(m^2)$"

    fmtr = ShorthandFormatter()
    a_str = fmtr.format("{0:.1u}",inv_params[0])
    sets.graph_label[1] = r"$a={}$".format(a_str)

    fig, ax = plot_multi_2(sets, x_values = [times_n, x_fit, [], []], y_values = [distances2_n, y_fit, [], []], x_err = [times_s,[],[],[]], y_err = [distances2_s,[],[],[]])
    #ax.fill_between(x_fit,y_fit_up,y_fit_down, facecolor = "y", alpha = 0.1)

if __name__ == "__main__":
    sets = get_settings()
    plot_line(sets)

    plt.show()