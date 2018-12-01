#-*- coding: utf8 -*-
import numpy as np
import scipy.odr.odrpack as odrpack
#from scipy.optimize import curve_fit
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
    general_sets.y_label = r"$<v> (m/s)$"

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

def linear(B,x):
    return B[0]*x+B[1]

def plot_line(sets):
    f = ft.linear2
    fit_samples = 10
    distances, times = get_data(sets)
    v_avg=distances/times
    distances2 = distances**2
    distances2_n, times_n, v_avg_n = unp.nominal_values(distances2), unp.nominal_values(times), unp.nominal_values(v_avg)
    distances2_s, times_s, v_avg_s = unp.std_devs(distances2), unp.std_devs(times), unp.std_devs(v_avg)
    print (distances2_s)
    
    #ODR fitting
    linear_model = odrpack.Model(linear)
    data = odrpack.RealData(distances2_n, v_avg_n, sx=distances2_s, sy=v_avg_s)
    myodr = odrpack.ODR(data,linear_model, beta0=[-1,1])
    output = myodr.run()
    params_n = output.beta
    params_s = output.sd_beta
    params = unp.uarray(params_n, params_s)

    x_fit = np.linspace(0, max(distances2_n), fit_samples)
    y_fit = linear(params_n, x_fit)
    y_fit_up = linear(params_n+params_s, x_fit)
    y_fit_down = linear(params_n-params_s, x_fit)

    #print (y_fit_up,y_fit_down, y_fit)
    fmtr = ShorthandFormatter()
    a_str = fmtr.format("{0:.1u}",params[0])
    v_str = fmtr.format("{0:.1u}",params[1])

    sets.graph_label[1] = "y= ax + v mit\n " + r"$a={}\ 1/ms$".format(a_str) + " und " + r"$v={}\ m/s$".format(v_str)
    fig, ax = plot_multi_2(sets, x_values = [distances2_n,x_fit,x_fit,x_fit], y_values = [v_avg_n,y_fit,y_fit_up,y_fit_down], x_err = [distances2_s,[],[],[]], y_err = [v_avg_s,[], [],[]])
    ax.fill_between(x_fit,y_fit_up,y_fit_down, facecolor = "y", alpha = 0.1)


    #x_fit = np.linspace(min(times_n), max(times_n), fit_samples)
    #y_fit = f(x_fit, *popt)
    #sets.x_label = "T (s)"
    #sets.y_label = r"$D^2 \:(m^2)$"
#
    #fmtr = ShorthandFormatter()
    #a_str = fmtr.format("{0:.1u}",inv_params[0])
    #sets.graph_label[1] = r"$a={}$".format(a_str)
#
    #fig, ax = plot_multi_2(sets, x_values = [times_n, x_fit, [], []], y_values = [distances2_n, y_fit, [], []], x_err = [times_s,[],[],[]], y_err = [distances2_s,[],[],[]])

if __name__ == "__main__":
    sets = get_settings()
    plot_line(sets)

    plt.show()