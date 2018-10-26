import matplotlib.pyplot as plt
import data_manager as dm
import math
import fit_functions as ft
from scipy.optimize import curve_fit

'''==========================================
                SETUP SECTION
=========================================='''
#find the csv file with the data
dataset_folder = "../datasets/"
dataset_file_name = "kirch2"
dataset_file_extension = ".csv"
save_plot = True #attention! overwrites old plot with same name
plot_folder = "../plots/"

include_legend = True
fit = True
include_parameter_values = True
add_origin = False
show_plot = True

#set the x and y data and errors
x_column_caption = "U0"
y_column_caption = "ratio"
x_err_column_caption = "fU0"
y_err_column_caption = "fratio"
#set axis labels and graph appearence
x_label = "Gesamtspannung [V]"
y_label = "U0/(U1+U2)"
graph_format = "bs" #respectively color and line style
axes_label_fontsize = "x-large"
error_bar_capsize = 2
#fit setup
function_to_fit = ft.constant
fit_graph_format = "r-"
fit_samples_number = 150
show_parameter_table = True
#legend setup
legend_location = "upper left"
legend_fontsize = "x-large"
raw_graph_label = ""
fitted_graph_label = "Gefitteter Verlauf"
#indexes of datapoints to discard
discard_datapoints_indexes = []

'''==========================================
=========================================='''



def frange(start, stop, step = 1.0):
    #returns a list like range, but with float
    i = start #froce it to be a float
    l = []
    st = abs(step)
    if stop>start:
        while i<=stop:
            l.append(i)
            i+=st
    elif stop<=start:
        while i>stop:
            l.append(i)
            i-=st
    return l


def f(x):
    #fitted f
    return function_to_fit(x,*popt) #*popt splits the array into single parameters

def plot (x_values, y_values, x_label, y_label, x_err = 0, y_err = 0):
    fig, ax = plt.subplots()
    ax.errorbar(x_values, y_values, xerr=x_err, yerr=y_err, capsize = error_bar_capsize, fmt = graph_format, label = raw_graph_label)
    plt.xlabel(x_label, fontsize = axes_label_fontsize)
    plt.ylabel(y_label, fontsize = axes_label_fontsize)
    
    if (fit):
        fit_start = x_values[0]
        fit_stop = x_values[-1]
        fit_step = (fit_stop-fit_start)/fit_samples_number
        x_fit = [i for i in frange(fit_start, fit_stop, fit_step)]
        y_fit = [f(i) for i in frange(fit_start, fit_stop, fit_step)]
        ax.plot (x_fit, y_fit, fit_graph_format, label = fitted_graph_label)

    if (include_legend):
        legend = ax.legend(loc = legend_location, fontsize = legend_fontsize)
    if (save_plot):
        #plot_file_name = plot_folder + dataset_file_name + x_column_caption + "-" + y_column_caption + ".png"
        plot_file_name = "{}{}-{}-{}.png".format(plot_folder, dataset_file_name, x_column_caption, y_column_caption)
        plt.savefig(plot_file_name)
    if (show_plot):
        plt.show()

'''
def plot (x_values, y_values, x_label, y_label, x_err = 0, y_err = 0):

    plt.errorbar(x_values,y_values, xerr=x_err, yerr=y_err, fmt = graph_format)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend("lalala")
        if (fit):
        fit_start = x_values[0]
        fit_stop = x_values[-1]
        fit_step = (fit_stop-fit_start)/fit_samples_number
        x_fit = [i for i in frange(fit_start, fit_stop, fit_step)]
        y_fit = [f(i) for i in frange(fit_start, fit_stop, fit_step)]
        plt.plot (x_fit, y_fit, fit_graph_format)
    plt.show()
'''

dataset = dm.csv_to_list(dataset_folder + dataset_file_name + dataset_file_extension)
x_values = dm.return_column (dataset, name = x_column_caption)
x_err_values = dm.return_column (dataset, name = x_err_column_caption)
y_values = dm.return_column (dataset, name = y_column_caption)
y_err_values = dm.return_column (dataset, name = y_err_column_caption)

discard_datapoints_indexes.sort(reverse = True) #necessary for removing correct point
for i in discard_datapoints_indexes:
    del x_values[i]
    del x_err_values[i]
    del y_values[i]
    del y_err_values[i]

popt, pcov = curve_fit(function_to_fit,x_values, y_values)
#ft.fit has scipy.optimize.curve_fit built in, with parameter bounding
#popt, pcov = ft.fit(function_to_fit,x_values, y_values)

if (add_origin):
    x_values.append(0.)
    x_err_values.append(0.)
    y_values.append(0.)
    y_err_values.append(0.)


if (fit):
    print (popt)
    print (pcov)
    plot_name = "{}-{}-{}".format(dataset_file_name, x_column_caption, y_column_caption)
    
    old_txt = open(plot_folder + "Parameters.txt", "r")
    old_content = old_txt.readlines()
    old_txt.close()
    
    param_txt = open(plot_folder + "Parameters.txt", "w")
    for i in old_content:
        param_txt.write(i)
    param_txt.write("\n" + plot_name + ": ")
    for p in popt:
        param_txt.write(str(p) + " ")
    param_txt.write ("\t  Covariance: ")
    for co in pcov:
        param_txt.write(str(co) + " ")
    param_txt.close()

plot(x_values, y_values, x_label, y_label, x_err_values, y_err_values)