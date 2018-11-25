import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.insert(0,"../../scripts")
import data_manager as dm
import fit_functions as ft
from myPlot import Settings, plot, plot_multi_2


def get_settings():
    general_sets = Settings()
    
    general_sets.dataset_folder = "../datasets/rest/"
    general_sets.x_column_caption = "TIME"
    general_sets.y_column_caption = "CH2"

    general_sets.x_label = "t (s)"
    general_sets.y_label = "Auslenkung (V)"

    general_sets.graph_format = ["b.","g.","r-","y--"]
    general_sets.fit_graph_format = "r-"
    general_sets.fitted_graph_label = r"$y = A\:\cos(\omega t + \phi)\: e^{-\beta t}$"
    general_sets.graph_label = ""
    general_sets.axes_label_fontsize = 20
    
    '''-------- Settings 1: stark gedampft--------'''
    sets1 = general_sets.clone()
    sets1.dataset_file_name = "T0000.csv"
    
    '''-------- Settings 2: schwach gedampft--------'''
    sets2 = general_sets.clone()
    sets2.dataset_file_name = "T0015.csv"

    '''-------- Settings 3: frei--------'''
    sets3 = general_sets.clone()
    #sets3.graph_format = "b."
    sets3.dataset_file_name = "T0029.csv"
    sets3.fitted_graph_label = r"$y=A\:\cos(\omega t + \phi)$"

    return sets1,sets2,sets3

def smooth(data,filter_length):
    #smooths a curve by taking the averave value of each neighboring data point in filter_length range
    return [np.average(data[i-filter_length:i+filter_length]) for i in range(filter_length,len(data)-filter_length)]

def get_data(sets):
    #reads data from datasets specified in Settings
    dataset = dm.csv_to_list(sets.dataset_folder + sets.dataset_file_name)
    time = dm.return_column (dataset, title = sets.x_column_caption, title_index = sets.title_index)
    deviation = dm.return_column (dataset, title = sets.y_column_caption, title_index = sets.title_index)
    return time, deviation

def plot_dampf_smooth(sets,filter_length):
    filter_length = 10
    f = ft.exponential_decay_oscillator
    fit_samples_number = 500

    time, deviation_raw = get_data(sets)
    raw_params, raw_cov = curve_fit(f,time,deviation_raw)
    print ("This is raw covariance: " + str(raw_cov))
    #sets.fitted_graph_label += "\n" + r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.9f}\:$".format(raw_params[1],raw_params[2]) + r"$s^{-1}$"
    offset = raw_params[-1]
    deviation = [d-offset for d in deviation_raw]
    raw_params[-1] = 0
    
    deviation_smooth = smooth(deviation,filter_length)
    time_smooth = time[filter_length:-filter_length]
    smooth_params, smooth_cov = curve_fit(f,time_smooth,deviation_smooth)
    
    x_fit = np.linspace(min(time), max(time), fit_samples_number)
    y_fit = [f(x,*raw_params) for x in x_fit]
    y_fit_smooth = f(x_fit, *smooth_params)
    print ("LENGTH OF YS: " + str(len(y_fit_smooth)) + " " + str(len(y_fit)))
    print ("This is smooth covariance: " + str(smooth_cov))
    sets.graph_label = ["","",
                        r"$y = A\:\cos(\omega t + \phi)\: e^{-\beta t}$" + "\n"
                        r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.9f}\:$".format(raw_params[1],raw_params[2]) + r"$s^{-1}$",
                        r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.9f}\:$".format(smooth_params[1],smooth_params[2]) + r"$s^{-1}$"]
    fig, ax = plot_multi_2(sets = sets, x_values = [time,time_smooth,x_fit,x_fit], y_values = [deviation,deviation_smooth,y_fit,y_fit_smooth])

def plot_dampf(sets):
    f = ft.exponential_decay_oscillator

    sets.graph_format = sets.graph_format[0]

    time, deviation_raw = get_data(sets)
    raw_params, raw_cov = curve_fit(f,time,deviation_raw)
    print ("This is raw covariance: " + str(raw_cov))
    #sets.fitted_graph_label += "\n" + r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.9f}\:$".format(raw_params[1],raw_params[2]) + r"$s^{-1}$"
    offset = raw_params[-1]
    deviation = [d-offset for d in deviation_raw]
    raw_params[-1] = 0

    fig, ax = plot(sets, time, deviation, f_to_fit = f, params = raw_params)

def plot_frei_smooth(sets,filter_length):
    f = ft.cos_phase
    fit_samples_number = 500

    time, deviation = get_data(sets)
    raw_params, raw_cov = curve_fit(f,time,deviation, p0 = [2,0.8,0])
    print ("This is raw covariance: " + str(raw_cov))
    #sets.fitted_graph_label += "\n" + r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.9f}\:$".format(raw_params[1],raw_params[2]) + r"$s^{-1}$"
    
    deviation_smooth = smooth(deviation,filter_length)
    time_smooth = time[filter_length:-filter_length]
    smooth_params, smooth_cov = curve_fit(f,time_smooth,deviation_smooth, p0 = [2,0.8,0])
    
    x_fit = np.linspace(min(time), max(time), fit_samples_number)
    y_fit = [f(x,*raw_params) for x in x_fit]
    y_fit_smooth = f(x_fit, *smooth_params)
    print ("LENGTH OF YS: " + str(len(y_fit_smooth)) + " " + str(len(y_fit)))
    print ("This is smooth covariance: " + str(smooth_cov))
    sets.graph_label = ["","",
                        r"$y = A\:\cos(\omega t + \phi)$" + "\n"
                        r"$\nu= 2\pi\omega = {:.4f} \:Hz$".format(raw_params[1]),
                        r"$\nu= 2\pi\omega = {:.4f} \:Hz$".format(smooth_params[1])]
    fig, ax = plot_multi_2(sets = sets, x_values = [time,time_smooth,x_fit,x_fit], y_values = [deviation,deviation_smooth,y_fit,y_fit_smooth])


def plot_frei(sets):
    f = ft.cos_phase
    time, deviation = get_data(sets)
    sets.graph_format = sets.graph_format[0]

    popt, cov = curve_fit(f, time, deviation, p0 = [2,0.8,0])
    print ("This is free covariance: " + str(cov))
    sets.fitted_graph_label += "\n" + r"$\nu= 2\pi\omega = {:.4f} \:Hz$".format(popt[1])
    fig, ax = plot(sets, time, deviation, f_to_fit = f, params = popt)

if __name__ == "__main__":
    smoothen = True
    filter_length = 20
    sets1, sets2, sets3 = get_settings()
    if (smoothen):
        plot_dampf_smooth(sets1,filter_length)
        plot_dampf_smooth(sets2,filter_length)
        plot_frei_smooth(sets3,filter_length)
    else:
        plot_dampf(sets1)
        plot_dampf(sets2)
        plot_frei(sets3)
    plt.show()
    
    
    
    '''
    filter_length = 20
    f = ft.exponential_decay_oscillator
    fit_samples_number = 500

    for i in range(2):
        time, deviation_raw = get_data(sets[i])
        raw_params, raw_cov = curve_fit(f,time,deviation_raw)
        print ("This is raw covariance: " + str(raw_cov))
        #sets[i].fitted_graph_label += "\n" + r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.9f}\:$".format(raw_params[1],raw_params[2]) + r"$s^{-1}$"
        offset = raw_params[-1]
        deviation = [d-offset for d in deviation_raw]
        raw_params[-1] = 0
        #fig, ax = plot(sets = sets[i], x_values = time, y_values = deviation, f_to_fit = f, params = popt)

        #smoothing check if beta is similar
        #deviation_smooth = smooth(deviation,filter_length)
        #time_smooth = time[filter_length:-filter_length]
        #popt, cov = popt, cov = curve_fit(f,time_smooth,deviation_smooth)
        #print ("This is smooth covariance: " + str(cov))
        #ax.plot(time_smooth, deviation_smooth,"g.", label = r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.9f}\:$".format(popt[1],popt[2]) + r"$s^{-1}$")
        #ax.legend()
        
        deviation_smooth = smooth(deviation,filter_length)
        time_smooth = time[filter_length:-filter_length]
        smooth_params, smooth_cov = curve_fit(f,time_smooth,deviation_smooth)
        
        x_fit = np.linspace(min(time), max(time), fit_samples_number)
        y_fit = [f(x,*raw_params) for x in x_fit]
        y_fit_smooth = f(x_fit, *smooth_params)
        print ("LENGTH OF YS: " + str(len(y_fit_smooth)) + str(len(y_fit)))
        print ("This is smooth covariance: " + str(smooth_cov))
        sets[i].graph_label = ["",
                            r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.9f}\:$".format(raw_params[1],raw_params[2]) + r"$s^{-1}$",
                            "",
                            r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.9f}\:$".format(smooth_params[1],smooth_params[2]) + r"$s^{-1}$"]
        fig, ax = plot_multi_2(sets = sets[i], x_values = [time,time_smooth,x_fit,x_fit], y_values = [deviation,deviation_smooth,y_fit,y_fit_smooth])

    time, deviation = get_data(sets[2])
    f = ft.cos_phase
    
    popt, cov = curve_fit(f, time, deviation, p0 = [2,0.8,0])
    print ("This is free covariance: " + str(cov))
    sets[2].fitted_graph_label += "\n" + r"$\nu= 2\pi\omega = {:.4f} \:Hz$".format(popt[1])
    fig, ax = plot(sets[2], time, deviation, f_to_fit = f, params = popt)
    '''