import numpy as np
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from os import listdir
import sys
sys.path.insert(0,"../../scripts")
import data_manager as dm
import fit_functions as ft


'''==========================================
                SETUP SECTION
=========================================='''
class Settings (object):
    subplots = 13
    dataset_folder = "../datasets/damp1/"
    dataset_file_names = listdir(dataset_folder)
    
    dataset_file_name = "wasser1"
    dataset_file_extension = ".csv"
    plot_folder = "../Versuch10/plots/"

    include_legend = True
    fit = True
    include_parameter_values = False
    add_origin = False
    save_plot = False #attention! overwrites old plot with same name
    show_plot = True

    #set the x and y data and errors
    title_index = 15
    x_column_caption = "h"
    y_column_caption = "h1"
    x_err_column_caption = "fh"
    y_err_column_caption = "fh1"
    include_error = False
    #set axis labels and graph appearence
    x_label = "Wasserhoehe davor dh (mm)"
    y_label = "Wasserhoehe danach dh' (mm)"
    graph_format = "" #respectively color and line style
    error_bar_capsize = 2
    axes_label_fontsize = 18
    axes_tick_fontsize = 18
    #fit setup
    #def get_function_to_fit(self):
    #    return ft.cos_phase
    #function_to_fit = ft.cos_phase
    fit_graph_format = "r-"
    fit_samples_number = 150
    show_parameter_table = True
    param_bounds = ([0,0,-180,-np.inf],[np.inf,np.inf,180,np.inf])
    #specific for damp1 and damp2 respectively
    initial_guesses_1 = [[2,0.8,60],[0.2,0.2,60],[0.5,0.4,90],[1,0.5,120],[2,0.6,60],[1.5,0.7,60],[2,0.8,60],[2,1,60],[0.5,1,60],[0.2,1.6,0],[0.2,1.6,60],[0.5,2,60],[0.15,2,-20]]
    initial_guesses_2 = [[2,0.8,60],[0.2,0.2,60],[0.5,0.4,90],[1,0.5,120],[1,0.6,60],[1.5,0.7,60],[2,0.8,60],[2,1,60],[0.5,1.1,60],[0.2,1.2,0],[0.3,1.3,180],[0.2,1.4,180],[0.15,1.5,-20]]

    #legend setup
    legend_location = "upper center"
    legend_fontsize = "x-large"
    raw_graph_label = ""
    fitted_graph_label = "Gefitteter Verlauf y = ax"#\na = 0.2278"
    #indexes of datapoints to discard
    discard_datapoints_indexes = []

'''==========================================
=========================================='''

def amplitude_frequency(w,w0,doj,b):
    return doj/(np.sqrt((w0**2-w**2)**2)+4*(b/2)**2*w**2)

def get_data(sets):
    #reads data from datasets specified in Settings
    datasets = [dm.csv_to_list(sets.dataset_folder + name) for name in sets.dataset_file_names]
    times = [dm.return_column (dataset, title = "TIME", title_index = 15) for dataset in datasets]
    forces = [dm.return_column (dataset, title = "CH1", title_index = 15) for dataset in datasets]
    deviations = [dm.return_column (dataset, title = "CH2", title_index = 15) for dataset in datasets]
    return datasets, times, forces, deviations

def plot(x,y):
    fig, ax = plt.subplots()
    ax.plot(x, y,"r.")

def plot_fit(sets,xs,ys,f,popts,covs):
    nr = int(math.sqrt(sets.subplots))
    nc = int(sets.subplots/nr)+1
    fig, axs = plt.subplots(nrows = nr, ncols = nc, squeeze=False)
    i=0
    for axr in axs:
        for ax in axr:
            try:
                ax.plot(xs[i], ys[i], sets.graph_format)
                fit_start = min(xs[i])
                fit_stop = max(xs[i])
                fit_step = (fit_stop-fit_start)/sets.fit_samples_number
                x_fit = np.arange(fit_start, fit_stop, fit_step)#frange(fit_start, fit_stop, fit_step)]
                y_fit = [f(x, *popts[i]) for x in x_fit]#[f(i) for i in frange(fit_start, fit_stop, fit_step)]
                ax.plot (x_fit, y_fit, sets.fit_graph_format, label ="{:.2f} Hz\n{:.2f} V".format(popts[i][1], abs(popts[i][0])))
                ax.legend(loc = sets.legend_location)
                i+=1
            except:
                print("boh")

def fit(sets,x,y,f):
    popts, covs = [],[]
    if sets.dataset_folder == "../datasets/damp1/":
        initial_guesses = sets.initial_guesses_1
    else:
        initial_guesses = sets.initial_guesses_2
    for i in range(len(sets.dataset_file_names)):
        popt, cov = curve_fit(f,x[i],y[i], p0 = initial_guesses[i]+[0])#, bounds = sets.param_bounds)
        popts.append(popt)
        covs.append(cov)
        print(popt[1])
    return popts, covs

if __name__ == "__main__":
    sets1 = Settings()
    sets2 = Settings()
    sets2.subplots = 1
    sets2.graph_format = "bo"
    sets2.legend_location = "upper left"
    f_to_fit_1 = ft.cos_phase_offset

    datasets, times, forces, deviations = get_data(sets1)
    popts, covs = fit(sets1,times,deviations,f_to_fit_1)
    plot_fit(sets1,times,deviations,f_to_fit_1,popts, covs)

    f_to_fit_2 = amplitude_frequency
    freqs = [popt[1] for popt in popts]
    amps = [abs(popt[0]) for popt in popts]
    popt, cov = curve_fit(f_to_fit_2, freqs, amps)
    #plot(freqs, amps)
    plot_fit(sets2, [freqs], [amps], f_to_fit_2, [popt], [cov])
    print ("BETA = " + str(popt[-1]))
    plt.show()