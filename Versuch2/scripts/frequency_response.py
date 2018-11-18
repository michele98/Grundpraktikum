import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from os import listdir

import sys
sys.path.insert(0,"../../scripts")
import data_manager as dm
import fit_functions as ft
from myPlot import Settings, plot, plot_multi


def get_settings():
    general_sets_multi = Settings()
    
    general_sets_multi.subplots_nrows = 3
    general_sets_multi.subplots_ncols = 5
    general_sets_multi.include_legend = False
        
    general_sets_multi.x_label = ""
    general_sets_multi.y_label = ""
    general_sets_multi.graph_format = "" #respectively color and line style
    general_sets_multi.error_bar_capsize = 0
    general_sets_multi.axes_label_fontsize = "medium"
    general_sets_multi.axes_tick_fontsize = "medium"
    
    general_sets_multi.include_legend = True
    general_sets_multi.raw_graph_label = ""
    general_sets_multi.legend_fontsize = "medium"
    general_sets_multi.legend_location = "upper left"

    '''----Settings 1----'''
    sets1 = general_sets_multi.clone()
    sets1.dataset_folder = "../datasets/damp1/"
    sets1.dataset_file_name = listdir(sets1.dataset_folder)
    
    '''----Settings 2----'''
    sets2 = general_sets_multi.clone()
    sets2.dataset_folder = "../datasets/damp2/"
    sets2.dataset_file_name = listdir(sets2.dataset_folder)

    '''----Settings 3 for frequency response----'''
    sets3 = general_sets_multi.clone()
    sets3.subplots_ncols = 1
    sets3.subplots_nrows = 1
    sets3.graph_format = "bs"
    sets3.fitted_graph_label = "Resonanzkurve"
    sets3.legend_fontsize = "xx-large"

    '''----Settings 4 for phase curve---'''
    sets4 =general_sets_multi.clone()
    sets4.subplots_ncols = 1
    sets4.subplots_nrows = 1
    sets4.graph_format = "bs"
    sets4.fitted_graph_label = "Phasenkurve"
    sets4.legend_fontsize = "xxlarge"

    return [sets1, sets2, sets3, sets4]

def amplitude_frequency(w,w0,doj,b):
    return doj/(np.sqrt((w0**2-w**2)**2)+4*(b/2)**2*w**2)

def phase_frequency():
    return 1


def get_data(sets):
    #reads data from datasets specified in Settings
    datasets = [dm.csv_to_list(sets.dataset_folder + name) for name in sets.dataset_file_name]
    times = [dm.return_column (dataset, title = "TIME", title_index = 15) for dataset in datasets]
    forces = [dm.return_column (dataset, title = "CH1", title_index = 15) for dataset in datasets]
    deviations = [dm.return_column (dataset, title = "CH2", title_index = 15) for dataset in datasets]
    return datasets, times, forces, deviations


if __name__ == "__main__":

    initial_guesses_1 = [[2,0.8,60],[0.2,0.2,60],[0.5,0.4,90],[1,0.5,120],[2,0.6,60],[1.5,0.7,60],[2,0.8,60],[2,1,60],[0.5,1,60],[0.2,1.6,0],[0.2,1.6,60],[0.5,2,60],[0.15,2,-20]]
    initial_guesses_2 = [[2,0.8,60],[0.2,0.2,60],[0.5,0.4,90],[1,0.5,120],[1,0.6,60],[1.5,0.7,60],[2,0.8,60],[2,1,60],[0.5,1.1,60],[0.2,1.2,0],[0.3,1.3,180],[0.2,1.4,180],[0.15,1.5,-20]]
    initial_guesses = [initial_guesses_1,initial_guesses_2]

    set_list = get_settings()
    f = ft.cos_phase_offset
    f2 = amplitude_frequency

    for i in range(2):
        datasets, times, forces, deviations = get_data(set_list[i])
        popts, covs = [],[]
        sets = []
        for j in range(len(set_list[i].dataset_file_name)):
            new_setting = set_list[i].clone()
            pzero = initial_guesses[i][j] + [0]
            popt, cov = curve_fit(f,times[j],deviations[j], p0 = pzero)#initial_guesses[i][j])#, bounds = sets.param_bounds)
            popts.append(popt)
            covs.append(cov)
            print(popt[1])
            new_setting.fitted_graph_label = "{:.2f} Hz".format(popt[1])
            sets.append(new_setting)
        fig, axs = plot_multi(sets,times,deviations, f_to_fit = f, params = popts)
        print(set_list[i].dataset_file_name)

        freqs = [popt[1] for popt in popts]
        amps = [abs(popt[0]) for popt in popts]
        popt, cov = curve_fit(f2, freqs, amps)
        #plot(freqs, amps)
        fig, ax = plot(set_list[2], freqs, amps, f_to_fit = f2, params = popt)
        print ("BETA = " + str(popt[-1]))



    plt.show()