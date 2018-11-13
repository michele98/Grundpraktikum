import matplotlib.pyplot as plt
import matplotlib.axes as axes
import data_manager as dm
import math
import fit_functions as ft
from scipy.optimize import curve_fit
from scipy.stats import chisquare

'''==========================================
                SETUP SECTION
=========================================='''
#find the csv file with the data
dataset_folder = "../Versuch10/datasets/"
dataset_file_name = "wasser2"
dataset_file_extension = ".csv"
plot_folder = "../Versuch10/plots/"

include_legend = True
fit = True
include_parameter_values = False
add_origin = False
save_plot = False #attention! overwrites old plot with same name
show_plot = True

#set the x and y data and errors
x_column_caption = "h"
y_column_caption = "h1"
x_err_column_caption = "fh"
y_err_column_caption = "fh1"
#set axis labels and graph appearence
x_label = "Wasserhoehe davor dh (mm)"
y_label = "Wasserhoehe danach dh' (mm)"
graph_format = "b." #respectively color and line style
error_bar_capsize = 2
axes_label_fontsize = 18
axes_tick_fontsize = 18
#fit setup
function_to_fit = ft.linear1
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
    ax.tick_params(labelsize = axes_tick_fontsize)
    plt.xlabel(x_label, fontsize = axes_label_fontsize)
    plt.ylabel(y_label, fontsize = axes_label_fontsize)
    plt.tight_layout() #makes room for larger label
    if (fit):
        fit_start = min(x_values)
        fit_stop = max(x_values)
        fit_step = (fit_stop-fit_start)/fit_samples_number
        x_fit = [i for i in frange(fit_start, fit_stop, fit_step)]
        y_fit = [f(i) for i in frange(fit_start, fit_stop, fit_step)]
        ax.plot (x_fit, y_fit, fit_graph_format, label = fitted_graph_label)
        print("Chi: " + str(chisquare(y_values, [f(i) for i in x_values])))
    if (include_legend):
        legend = ax.legend(loc = legend_location, fontsize = legend_fontsize)
    if (save_plot):
        #plot_file_name = plot_folder + dataset_file_name + x_column_caption + "-" + y_column_caption + ".png"
        plot_file_name = "{}{}-{}-{}-{}.png".format(plot_folder, dataset_file_name,function_to_fit.__name__, x_column_caption, y_column_caption)
        plt.savefig(plot_file_name)#, bbox_inches ="tight")
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

if (add_origin):
    x_values.append(0.)
    x_err_values.append(0.)
    y_values.append(0.)
    y_err_values.append(0.)

#popt, pcov = curve_fit(function_to_fit,x_values, y_values, sigma = y_err_values, absolute_sigma=True)
popt, pcov = curve_fit(function_to_fit,x_values, y_values, sigma = y_err_values, absolute_sigma=True)
#ft.fit has scipy.optimize.curve_fit built in, with parameter bounding
#popt, pcov = ft.fit(function_to_fit,x_values, y_values)

if (fit):
    print (popt)
    print (pcov)
    plot_name = "{}-{}-{}".format(dataset_file_name, x_column_caption, y_column_caption)
    
    if (include_parameter_values):
        #saves old content of txt file
        old_txt = open(plot_folder + "Parameters.txt", "r")
        old_content = old_txt.readlines()
        old_txt.close()
        
        param_txt = open(plot_folder + "Parameters.txt", "w")
        for i in old_content: #rewrites old conent
            param_txt.write(i)
        #param_txt.write("\n" + plot_name + ": ") #writes dataset-xAxis-yAxis
        param_txt.write("\n{}; X={}, Y={}; ".format(dataset_file_name, x_column_caption, y_column_caption))
        param_txt.write("fit={}; ".format(function_to_fit.__name__))
        param_txt.write("Parameters= ")
        for p in popt: #writes parameter values
            param_txt.write(str(p) + " ")
        param_txt.write ("; Covariance: ")
        for co in pcov: #writes covariance matrix
            param_txt.write(str(co) + " ")
        param_txt.write("; K = {}".format(str(1/(1-popt[0])))) #calculates K from steigung
        #param_txt.write("; K = "+ str(popt[0])) #calculates K from steigung
        param_txt.close()

plot(x_values, y_values, x_label, y_label, x_err_values, y_err_values)