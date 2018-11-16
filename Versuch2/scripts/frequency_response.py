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
dataset_folder = "../datasets/rest/"
dataset_file_names = listdir(dataset_folder)

function_to_fit = ft.exponential_decay_oscillator
fit_graph_format = "r-"
fit_samples_number = 150
show_parameter_table = True

#legend setup
legend_location = "upper left"
legend_fontsize = "x-large"
raw_graph_label = ""
fitted_graph_label = "Gefitteter Verlauf y = ax"#\na = 0.2278"
#indexes of datapoints to discard
discard_datapoints_indexes = []

'''==========================================
=========================================='''

datasets = [dm.csv_to_list(dataset_folder + name) for name in dataset_file_names]

times = [dm.return_column (dataset, title = "TIME", title_index = 15) for dataset in datasets]
forces = [dm.return_column (dataset, title = "CH1", title_index = 15) for dataset in datasets]
deviations = [dm.return_column (dataset, title = "CH2", title_index = 15) for dataset in datasets]


def plot():
    nr = int(math.sqrt(len(dataset_file_names)))
    nc = int(len(dataset_file_names)/nr)
    fig, axs = plt.subplots(nrows = nr, ncols = nc)
    i=0
    popts, covs = fit()
    for axr in axs:
        for ax in axr:
            try:
                print (i)
                ax.plot(times[i], deviations[i])
                fit_start = min(times[i])
                fit_stop = max(times[i])
                fit_step = (fit_stop-fit_start)/fit_samples_number
                x_fit = np.arange(fit_start, fit_stop, fit_step)#frange(fit_start, fit_stop, fit_step)]
                y_fit = [function_to_fit(x, *popts[i]) for x in x_fit]#[f(i) for i in frange(fit_start, fit_stop, fit_step)]
                ax.plot (x_fit, y_fit, fit_graph_format, label = fitted_graph_label)
                i+=1
            except:
                print("boh")
    plt.show()

def fit():
    popts, covs = [],[]
    for i in range(len(dataset_file_names)):
        popt, cov = curve_fit(function_to_fit, times[i], deviations[i])
        popts.append(popt)
        covs.append(cov)
        print(popt[-1])
    return popts, covs

print (np.shape(datasets))
print (dataset_file_names)
popts, covs = plot()