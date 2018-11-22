import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,"../../scripts")
import data_manager as dm
import fit_functions as ft
from myPlot import Settings, plot


def get_settings():
    general_sets = Settings()

    general_sets.x_column_caption = "TIME"
    general_sets.y_column_caption = "CH2"

    general_sets.x_label = "t (s)"
    general_sets.y_label = "Auslenkung (V)"

    general_sets.graph_format = "b."
    general_sets.fit_graph_format = "r-"
    general_sets.fitted_graph_label = r"$y = A\:\cos(\omega t + \phi)\: e^{-\beta t}$"
    general_sets.raw_graph_label = ""
    general_sets.axes_label_fontsize = 20
    
    '''-------- Settings 1: stark gedampft--------'''
    sets1 = general_sets.clone()
    sets1.dataset_folder = "../datasets/rest/"
    sets1.dataset_file_name = "T0015.csv"
    
    '''-------- Settings 2: schwach gedampft--------'''
    sets2 = general_sets.clone()
    sets2.dataset_folder = "../datasets/rest/"
    sets2.dataset_file_name = "T0000.csv"

    '''-------- Settings 3: frei--------'''
    sets3 = general_sets.clone()
    sets3.dataset_folder = "../datasets/rest/"
    sets3.dataset_file_name = "T0001.csv"

    sets3.fitted_graph_label = r"$y=A\:\cos(\omega t + \phi)$"
    
    return [sets1,sets2,sets3]

def get_data(sets):
    #reads data from datasets specified in Settings
    dataset = dm.csv_to_list(sets.dataset_folder + sets.dataset_file_name)
    time = dm.return_column (dataset, title = sets.x_column_caption, title_index = sets.title_index)
    deviation = dm.return_column (dataset, title = sets.y_column_caption, title_index = sets.title_index)
    return time, deviation

if __name__ == "__main__":
    sets = get_settings()

    for i in range(2):
        time, deviation_raw = get_data(sets[i])
        f = ft.exponential_decay_oscillator
        popt, cov = curve_fit(f,time,deviation_raw)
        sets[i].fitted_graph_label += "\n" + r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {}\:$".format(popt[1],popt[2]) + r"$s^{-1}$"
        offset = popt[-1]
        deviation = [d-offset for d in deviation_raw]
        popt[-1] = 0
        fig, ax = plot(sets = sets[i], x_values = time, y_values = deviation, f_to_fit = f, params = popt)
    
    time, deviation = get_data(sets[2])
    f = ft.cos_phase
    popt, pcov = curve_fit(f, time, deviation, p0 = [2,0.8,0])
    sets[2].fitted_graph_label += "\n" + r"$\nu= 2\pi\omega = {:.9f} \:Hz$".format(popt[1])
    fig, ax = plot(sets[2], time, deviation, f_to_fit = f, params = popt)
    plt.show()