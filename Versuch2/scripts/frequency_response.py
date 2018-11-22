import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from os import listdir

import sys
sys.path.insert(0,"../../scripts")
import data_manager as dm
import fit_functions as ft
from myPlot import Settings, plot, plot_multi, plot_subplots


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
    general_sets_multi.graph_label = ""
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
    sets3.graph_format = ["bs", "g-"]
    sets3.fitted_graph_label = "Resonanzkurve"
    sets3.graph_label = ["", "Theoretische Kurve"]
    sets3.legend_fontsize = "xx-large"

    '''----Settings 4 for phase curve---'''
    sets4 =general_sets_multi.clone()
    sets4.subplots_ncols = 1
    sets4.subplots_nrows = 1
    sets4.graph_format = ["bs","g-"]
    sets4.fitted_graph_label = "Phasenkurve"
    sets4.legend_fontsize = "xxlarge"

    return [sets1, sets2, sets3, sets4]

def amplitude_frequency(w,w0,doj,b):
    return doj/(np.sqrt((w0**2-w**2)**2)+4*(b/2)**2*w**2)

def V(A0,n,d):
    return A0/(np.sqrt((1-n**2)**2+4*d**2*n**2))

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

    initial_guesses_forces = [[2,0.8,60],[2,0.3,90],[2,0.4,90],[2,0.5,120],[2,0.6,60],[2,0.7,60],[2,0.8,60],[2,1,60],[2,1,60],[0.2,1.6,0],[0.2,1.6,60],[0.5,2,60],[0.15,2,-20]]
    freq0s  = [0.859619415952, 0.857223684731]
    betas = [0.276223958059, 0.507317117254]
    dojs = [5,9]

    set_list = get_settings()
    f = ft.cos_phase_offset
    f2 = amplitude_frequency

    for i in range(2):
        datasets, times, forces, deviations = get_data(set_list[i])
        oscillations_params_list, covs = [],[]
        forces_params_list = []
        
        sets = []
        sets_force = []
        # plots all wave forms for each anregung
        for j in range(len(set_list[i].dataset_file_name)):
            new_setting = set_list[i].clone()

            pzero = initial_guesses[i][j] + [0]
            popt_o, cov = curve_fit(f,times[j],deviations[j], p0 = pzero)
            oscillations_params_list.append(popt_o)
            
            new_setting.fitted_graph_label = "{:.2f} Hz".format(popt_o[1])
            sets.append(new_setting)
        fig, axs = plot_subplots(sets,times,deviations, f_to_fit = f, params = oscillations_params_list)

        #forces plotting
        for j in range(len(set_list[i].dataset_file_name)):
            new_setting_force = set_list[i].clone()

            popt, cov = curve_fit(f, times[j], forces[j], p0 = initial_guesses_forces[i] + [0])
            forces_params_list.append(popt)
            new_setting_force.fitted_graph_label = "{:.2f} Hz".format(popt[1])
            sets_force.append(new_setting_force)
        
        fig, axs = plot_subplots(sets_force,times,forces, f_to_fit = f, params = forces_params_list)
        print(set_list[i].dataset_file_name)
        
        #amplitude frequency response
        freqs = np.array([popt[1] for popt in oscillations_params_list])
        amps = np.array([abs(popt[0]) for popt in oscillations_params_list])
        frequency_response_params, cov = curve_fit(amplitude_frequency, freqs, amps)
        freqs2 = freqs*2
        set_list[2].fitted_graph_label = "Resonanzkurve\nberechnetes " + r"$\beta = {:.4f}$".format(frequency_response_params[-1]) + "\ntheoretisches " + r"$\beta = {:.4f}$".format(betas[i])

        xt = np.linspace(min(freqs), max(freqs), 150)
        theoretical_amps = [amplitude_frequency(2*np.pi*freq, 2*np.pi*freq0s[i], dojs[i], betas[i]) for freq in xt]
        x = [freqs, xt]
        y = [amps, theoretical_amps]
        #fig, ax = plot_multi(set_list[2], x, y, f_to_fit = amplitude_frequency, params = frequency_response_params)
        
        phase_response = np.array([popt[2] for popt in oscillations_params_list])
        
    plt.show()