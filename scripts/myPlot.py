import matplotlib.pyplot as plt
import matplotlib.axes as axes
import data_manager as dm
import fit_functions as ft
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import numpy as np
import copy
import uncertainties
import string

'''zum fitten gut: lmfit'''

'''==========================================
                SETUP SECTION
=========================================='''
class Settings(object):
    '''======================================
    defines how one subplot should look like.
    For multiple subplots, multiple settings objects are needed
    ========================================'''
    #find the csv file with the data
    dataset_folder = "../Versuch2/datasets/rest/"
    dataset_file_name = "T0015.csv"     #can be a list if multiple files are needed
    plot_folder = "../Versuch2/plots/"

    include_legend = True
    fit = True
    include_parameter_values = False
    add_origin = False
    save_plot = False #attention! overwrites old plot with same name
    show_plot = True
    subplots_nrows = 1
    subplots_ncols = 1

    #set the x and y data and errors, can be lists if multiple funcitons are in one plot
    title_index = 15
    x_column_caption = "TIME"
    y_column_caption = "CH2"
    x_err_column_caption = "fh"
    y_err_column_caption = "fh1"
    #include_error = False

    #set axis labels and graph appearence
    x_label = "X"
    y_label = "Y"
    title = ''
    title_fontsize = 14
    graph_format = "b." #respectively color and line style, can be list
    error_bar_capsize = 2
    axes_label_fontsize = 18
    axes_tick_fontsize = 18

    #fit setup (obsolete)
    def get_function_to_fit(self):
        return ft.cos_phase
    function_to_fit = ft.exponential_decay_oscillator
    fit_graph_format = "r-"
    fit_samples_number = 500
    show_parameter_table = True
    param_bounds = ([0,0,-180,-np.inf],[np.inf,np.inf,180,np.inf])
    #specific for damp1 and damp2 respectively
    initial_guesses_1 = [[2,0.8,60],[0.2,0.2,60],[0.5,0.4,90],[1,0.5,120],[2,0.6,60],[1.5,0.7,60],[2,0.8,60],[2,1,60],[0.5,1,60],[0.2,1.6,0],[0.2,1.6,60],[0.5,2,60],[0.15,2,-20]]
    initial_guesses_2 = [[2,0.8,60],[0.2,0.2,60],[0.5,0.4,90],[1,0.5,120],[1,0.6,60],[1.5,0.7,60],[2,0.8,60],[2,1,60],[0.5,1.1,60],[0.2,1.2,0],[0.3,1.3,180],[0.2,1.4,180],[0.15,1.5,-20]]
    
    #legend setup
    legend_location = "upper left"
    legend_fontsize = "x-large"
    graph_label = "Datenpunkte" #can be list
    fitted_graph_label = ""#\na = 0.2278" (obsolete)
    #indexes of datapoints to discard (obsolete)
    discard_datapoints_indexes = []

    def clone(self):
        return copy.deepcopy(self)

    def log(self, txt_path, function_to_fit, popt, pcov, value_dict):
        '''to finish!!!'''
        #saves old content of txt file
        old_txt = open(txt_path, "r")
        old_content = old_txt.readlines()
        old_txt.close()
        
        param_txt = open(txt_path, "w")
        for i in old_content: #rewrites old conent
            param_txt.write(i)
        #param_txt.write("\n" + plot_name + ": ") #writes dataset-xAxis-yAxis
        param_txt.write("\n{}; X={}, Y={}; ".format(self.dataset_file_name, self.x_column_caption, y_column_caption))
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

'''==========================================
=========================================='''

class ShorthandFormatter(string.Formatter):

    def format_field(self, value, format_spec):
        if isinstance(value, uncertainties.UFloat):
            return value.format(format_spec+'S')  # Shorthand option added
        # Special formatting for other types can be added here (floats, etc.)
        else:
            # Usual formatting:
            return super(ShorthandFormatter, self).format_field(
                value, format_spec)


def plot_multi (sets,x_values, y_values, x_err = [], y_err = [], f_to_fit = None, params = None):
    fig, ax = plt.subplots()

    for i in range(len(x_values)):
        if (len(x_err) != 0 or len(y_err) != 0):#if (sets.include_error):
            ax.errorbar(x_values[i], y_values[i], xerr=x_err[i], yerr=y_err[i], capsize = sets.error_bar_capsize, fmt = sets.graph_format[i], label = sets.graph_label[i])
        else:
             ax.plot(x_values[i], y_values[i], sets.graph_format[i], label = sets.graph_label[i])
        
        ax.tick_params(labelsize = sets.axes_tick_fontsize)
        #ax.set_xlim(0,3)
        plt.xlabel(sets.x_label, fontsize = sets.axes_label_fontsize)
        plt.ylabel(sets.y_label, fontsize = sets.axes_label_fontsize)
        plt.tight_layout() #makes room for larger label
        
    if(f_to_fit != None):
        fit_start = min(x_values[0])
        fit_stop = max(x_values[0])
        fit_step = (fit_stop-fit_start)/sets.fit_samples_number
        x_fit = np.arange(fit_start,fit_stop,fit_step)
        y_fit = [f_to_fit(x, *params) for x in x_fit]
        ax.plot(x_fit, y_fit, sets.fit_graph_format, label = sets.fitted_graph_label)
        #print("Chi: " + str(chisquare(y_values, [f_to_fit(i,*params) for i in x_values])))

    if (sets.include_legend):
        legend = ax.legend(loc = sets.legend_location, fontsize = sets.legend_fontsize)

    if (sets.save_plot):
        #plot_file_name = plot_folder + dataset_file_name + x_column_caption + "-" + y_column_caption + ".png"
        plot_file_name = "{}{}-{}-{}-{}.png".format(sets.plot_folder, sets.dataset_file_name,f_to_fit.__name__, sets.x_column_caption, sets.y_column_caption)
        plt.savefig(plot_file_name)#, bbox_inches ="tight")
    
    return fig, ax

'''=========
    preferred function to use!!!
========'''
def plot_multi_2 (sets,x_values, y_values, x_err = [[]], y_err = [[]]):
    fig, ax = plt.subplots()

    for i in range(len(x_values)):
        try:
            errorbar = len(x_err[i]) !=0 or len(y_err[i]) !=0
        except:
            errorbar = len(x_err[0]) !=0 or len(y_err[0]) !=0
        if (errorbar):#if (sets.include_error):
            ax.errorbar(x_values[i], y_values[i], xerr=x_err[i], yerr=y_err[i], capsize = sets.error_bar_capsize, fmt = sets.graph_format[i], label = sets.graph_label[i])
            #ax.set_ylim([min(y_values[i])-max(y_values[i])*0.05, max(y_values[i])*1.2])
        else:
            ax.plot(x_values[i], y_values[i], sets.graph_format[i], label = sets.graph_label[i])
        
        ax.tick_params(labelsize = sets.axes_tick_fontsize)
        ax.set_title(sets.title, fontsize = sets.title_fontsize)
        #ax.set_xlim(0,3)
        plt.xlabel(sets.x_label, fontsize = sets.axes_label_fontsize)
        plt.ylabel(sets.y_label, fontsize = sets.axes_label_fontsize)
        plt.tight_layout() #makes room for larger label
        
    if (sets.include_legend):
        legend = ax.legend(loc = sets.legend_location, fontsize = sets.legend_fontsize)

    if (sets.save_plot):
        #plot_file_name = plot_folder + dataset_file_name + x_column_caption + "-" + y_column_caption + ".png"
        #plot_file_name = "{}{}-{}-{}-{}.png".format(sets.plot_folder, sets.dataset_file_name,f_to_fit.__name__, sets.x_column_caption, sets.y_column_caption)
        plot_file_name = "plot.png"
        plt.savefig(plot_file_name)#, bbox_inches ="tight")
    
    return fig, ax

def plot (sets,x_values, y_values, x_err = [], y_err = [], f_to_fit = None, params = None):
    fig, ax = plt.subplots()
    
    if (len(x_err) != 0 or len(y_err) != 0):#if (sets.include_error):
        ax.errorbar(x_values, y_values, xerr=x_err, yerr=y_err, capsize = sets.error_bar_capsize, fmt = sets.graph_format, label = sets.graph_label)
    else:
        ax.plot(x_values, y_values, sets.graph_format, label = sets.graph_label)

    
    ax.tick_params(labelsize = sets.axes_tick_fontsize)
    #ax.set_xlim(0,3)
    plt.xlabel(sets.x_label, fontsize = sets.axes_label_fontsize)
    plt.ylabel(sets.y_label, fontsize = sets.axes_label_fontsize)
    plt.tight_layout() #makes room for larger label
    
    if(f_to_fit != None):
        fit_start = min(x_values)
        fit_stop = max(x_values)
        fit_step = (fit_stop-fit_start)/sets.fit_samples_number
        x_fit = np.arange(fit_start,fit_stop,fit_step)
        y_fit = [f_to_fit(x, *params) for x in x_fit]
        ax.plot(x_fit, y_fit, sets.fit_graph_format, label = sets.fitted_graph_label)
        #print("Chi: " + str(chisquare(y_values, [f_to_fit(i,*params) for i in x_values])))

    if (sets.include_legend):
        legend = ax.legend(loc = sets.legend_location, fontsize = sets.legend_fontsize)

    if (sets.save_plot):
        #plot_file_name = plot_folder + dataset_file_name + x_column_caption + "-" + y_column_caption + ".png"
        plot_file_name = "{}{}-{}-{}-{}.png".format(sets.plot_folder, sets.dataset_file_name,f_to_fit.__name__, sets.x_column_caption, sets.y_column_caption)
        plt.savefig(plot_file_name)#, bbox_inches ="tight")
    
    return fig, ax

def plot_subplots (sets,x_values, y_values, x_err = [], y_err = [], f_to_fit = None, params = None):
    j = 0
    fig, axs = plt.subplots(nrows = sets[j].subplots_nrows, ncols = sets[j].subplots_ncols, squeeze = False)
    print (np.shape(axs))
    for axr in axs:
        for ax in axr:
            try:
                if (len(x_err) != 0 or len(y_err) != 0):#if (sets[j].include_error):
                    ax.errorbar(x_values[j], y_values[j], xerr=x_err[j], yerr=y_err[j], capsize = sets[j].error_bar_capsize, fmt = sets[j].graph_format, label = sets[j].graph_label)
                else:
                    ax.plot(x_values[j], y_values[j], sets[j].graph_format, label = sets[j].graph_label)
                ax.tick_params(labelsize = sets[j].axes_tick_fontsize)
                plt.xlabel(sets[j].x_label, fontsize = sets[j].axes_label_fontsize)
                plt.ylabel(sets[j].y_label, fontsize = sets[j].axes_label_fontsize)
                #plt.tight_layout() #makes room for larger label
                if(f_to_fit != None):
                    fit_start = min(x_values[j])
                    fit_stop = max(x_values[j])
                    fit_step = (fit_stop-fit_start)/sets[j].fit_samples_number
                    x_fit = np.arange(fit_start,fit_stop,fit_step)
                    y_fit = [f_to_fit(x, *params[j]) for x in x_fit]
                    ax.plot(x_fit, y_fit, sets[j].fit_graph_format, label = sets[j].fitted_graph_label)
                    #print("Chi: " + str(chisquare(y_values, [f_to_fit(i,*params) for i in x_values])))
                if (sets[j].include_legend):
                    legend = ax.legend(loc = sets[j].legend_location, fontsize = sets[j].legend_fontsize)
                
                j+=1
            except:
                print("Empty subplot")
    if (sets[0].save_plot):
            #plot_file_name = plot_folder + dataset_file_name + x_column_caption + "-" + y_column_caption + ".png"
            plot_file_name = "{}{}-{}-{}-{}.png".format(sets[0].plot_folder, sets[0].dataset_file_name,f_to_fit.__name__, sets[0].x_column_caption, sets[0].y_column_caption)
            plt.savefig(plot_file_name)#, bbox_inches ="tight")
    return fig, axs

def error_string(value, error):
    #currently works only with errors<1
    value_l = list(str(value))
    error_l = list(str(error))

    if (len(value_l)<len(error_l)):
        value_l += ['0' for i in range(len(error_l)-len(value_l))]
    #print ("THIS IS ERROR IN: " + str(error))
    
    index_v, index_e = 0,0
    #part1
    try:
        if error_l[-4] == 'e':
            #index = 1 + int("".join(error_l[-2:]))
            ex = int("".join(error_l[-2:]))
            additional_values = ['0']+['.']+['0' for i in range(ex-1)]
            error_l = additional_values + error_l
            del(error_l[-4:])
    except: print ("Boh")
    
    for i in range(len(error_l)):
        if error_l[i]!='0' and error_l[i]!='.':
            index_e, index_v = i,i
            if(value_l[1] == '.'):
                index_v += 1
            break
    
    try:
        if int(value_l[index_e + 1]) >= 5:
            value_l[index_e] = str(int(value_l[index_e]) + 1)
    except: print("boh")
    try:
        if int(error_l[index_e + 1]) >= 5:
            error_l[index_e] = str(int(error_l[index_e]) + 1)
    except: print("booh")

    value_l = value_l[:index_v]
    value_l.append('(' + str(error_l[index_e]) + ')')

    return "".join(value_l)