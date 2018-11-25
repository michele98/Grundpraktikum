import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from os import listdir
import pandas

import sys
sys.path.insert(0,"../../scripts")
import data_manager as dm
import fit_functions as ft
from myPlot import Settings, plot, plot_multi_2, plot_subplots


def get_settings():
    general_sets_multi = Settings()
    

    general_sets = Settings()
    
    general_sets.dataset_folder = "../datasets/rest/"
    #general_sets.x_column_caption = "TIME"
    #general_sets.y_column_caption = "CH2"

    general_sets.x_label = r"$\eta$"
    general_sets.y_label = ["V", r"$\alpha\:(deg)$"]

    general_sets.graph_format = ["b.","g-","r-","y--"]
    general_sets.fit_graph_format = "r-"
    general_sets.fitted_graph_label = "Gefittete Kurve: " + r"$y = A\:\cos(\omega t + \phi)\: e^{-\beta t}$"
    general_sets.graph_label = ["Resonanzkurve", "Phasenverschiebung"]
    general_sets.axes_label_fontsize = 20
    
    '''----Settings 1: schwach gedampft Amplitude----'''
    sets1 = general_sets.clone()
    sets1.dataset_file_name = "frequency_response_1.csv"
    sets1.graph_label = general_sets.graph_label[0]
    sets1.y_label = general_sets.y_label[0]

    '''----Settings 2: stark gedampft Amplitude----'''
    sets2 = general_sets.clone()
    sets2.dataset_file_name = "frequency_response_2.csv"
    sets2.graph_label = general_sets.graph_label[0]
    sets2.y_label = general_sets.y_label[0]

    '''----Settings 3: schwach gedampft Phase----'''
    sets3 = sets1.clone()
    sets3.graph_label = general_sets.graph_label[1]
    sets3.y_label = general_sets.y_label[1]

    '''----Settings 4: stark gedampft Phase---'''
    sets4 = sets2.clone()
    sets4.graph_label = general_sets.graph_label[1]
    sets4.y_label = general_sets.y_label[1]

    return [sets1, sets2, sets3, sets4]

def amplitude_frequency(n,d):
    return 1/np.sqrt((1-n**2)**2+4*(d*n)**2)

def amplitude_frequency_non_normalized(n,d,A0):
    return A0/np.sqrt((1-n**2)**2+4*(d*n)**2)

def amplitude_frequency_boh(w,beta,A0,w0):
    return A0/(np.sqrt((w0**2-w**2)**2)+4*beta**2*w**2)
    #return A0*w0**2/(np.sqrt((w0**2-w**2)**2)+4*beta**2*w**2)

def phase_frequency(n,d):
    args = 2*d*n/(1-n**2)
    ret = np.zeros(len(n))
    for i in range(len(n)):
        if n[i]<1:
            ret[i] = np.arctan(args[i])*180/np.pi
        else:
            ret[i] = (np.arctan(args[i])+np.pi)*180/np.pi
    return ret
    #return np.arctan(2*d*n/(1-n**2))*180/np.pi

def get_data(sets):
    #reads data from datasets specified in Settings
    dataset = dm.csv_to_list(sets.dataset_folder + sets.dataset_file_name)
    
    freqeuncy_vals = dm.return_column (dataset, title = "FREQUENCY")
    frequency_err =dm.return_column(dataset, title = "FREQUENCY ERROR")
    frequency = [freqeuncy_vals, frequency_err]

    amplitude_vals = dm.return_column (dataset, title = "AMPLITUDE")
    amplitude_err = dm.return_column (dataset, title = "AMPLITUDE ERROR")
    amplitude = [amplitude_vals, amplitude_err]

    phase_difference_vals = dm.return_column (dataset, title = "PHASE DIFFERENCE")
    phase_difference_err = dm.return_column (dataset, title = "PHASE DIFFERENCEE ERROR")
    phase_difference = [phase_difference_vals,phase_difference_err]

    return frequency, amplitude, phase_difference

def plot_amplitude(sets, beta, f0):
    curve_samples_number = 500
    #f = amplitude_frequency_non_normalized
    f = amplitude_frequency_boh
    frequency, amplitude, phase_difference = get_data(sets) #each is 2D list with values and error
    #sets.graph_format = sets.graph_format[0]
    
    omega0 = f0*2*np.pi
    d = beta/omega0
    n = np.array(frequency)/f0
    frequency_axis = np.linspace(min(frequency[0]), max(frequency[0]), curve_samples_number)
    n_axis = frequency_axis/f0 #np.linspace(min(n[0]), max(n[0]), curve_samples_number)
    #y_theoretical = f(n_axis,np.zeros(len(n_axis)) + d, 1)
    y_theoretical = amplitude_frequency(n_axis, np.zeros(len(n_axis)) + d)
    
    params, cov = curve_fit(f,frequency[0],amplitude[0], sigma = amplitude[1], absolute_sigma = True)
    omega0 = params[2]
    A0 = params[1]/omega0**2
    f0_fit = params[2]/(2*np.pi)
    x_fit = frequency_axis/f0_fit
    y_fit = f(frequency_axis, *params)/A0

    #v = np.array(amplitude)/(amplitude[0][1])
    v = np.array(amplitude)/A0

    #print ("d = {}".format(params[0]))


    sets.graph_label = ["Aufnahmen mit beta = {:.3f}".format(beta),"gegebene beta un omega0","Fit mit\n" + r"$\omega_0 = {}\ \beta = {}$".format(params[2], params[1])]

    fig, ax = plot_multi_2(sets, x_values = [n[0], n_axis, x_fit], y_values = [v[0], y_theoretical, y_fit],
            x_err = [n[1], [], []], y_err = [v[1], [], []])


def plot_phase(sets, beta, f0):
    curve_samples_number = 500
    f = phase_frequency
    frequency, amplitude, phase_difference = get_data(sets) #each is 2D list with values and error
    #sets.graph_format = sets.graph_format[0]
    
    omega0 = f0*2*np.pi
    d = beta/omega0
    n = np.array(frequency)/f0
    print (n)
    n_axis = np.linspace(min(n[0]), max(n[0]), curve_samples_number)
    y_theoretical = f(n_axis,np.zeros(len(n_axis)) + d)
    
    params, cov = curve_fit(f,n[0],phase_difference[0], sigma = phase_difference[1], absolute_sigma = True)
    y_fit = f(n_axis, *params)
    print ("d = {}".format(params[0]))

    sets.graph_label = ["Aufnahmen mit Beta = {:.3f}".format(beta),"gegebene beta un omega0","Fit"]

    fig, ax = plot_multi_2(sets, x_values = [n[0], n_axis, n_axis], y_values = [phase_difference[0], y_theoretical, y_fit],
            x_err = [n[1], [], []], y_err = [phase_difference[1], [], []])


if __name__ == "__main__":
    sets_list = get_settings()
    betas = np.array([0.27622, 0.507317])
    f0 = 0.85997

    print ("Gegebene: d = {}, {}".format(betas[0]/(2*np.pi*f0),betas[1]/(2*np.pi*f0)))

    for i in range(2):
        plot_amplitude(sets_list[i], betas[i],f0)
        #plot_phase(sets_list[i+2], betas[i], f0)

    plt.show()