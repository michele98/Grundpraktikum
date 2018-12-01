import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties.umath import *

import sys
sys.path.insert(0,"../../scripts")
import data_manager as dm
import fit_functions as ft
from myPlot import Settings, plot, plot_multi_2, error_string, ShorthandFormatter


def get_settings():
    general_sets = Settings()
    
    general_sets.dataset_folder = "../datasets/rest/"
    #general_sets.x_column_caption = "TIME"
    #general_sets.y_column_caption = "CH2"

    general_sets.x_label = "t (s)"
    general_sets.y_label = "Auslenkung (V)"

    general_sets.graph_format = ["b.","g.","r-","y--"]
    general_sets.fit_graph_format = "r-"
    general_sets.fitted_graph_label = "Gefittete Kurve: " + r"$y = A\:\cos(\omega t + \phi)\: e^{-\beta t}$"
    general_sets.graph_label = ""
    general_sets.axes_label_fontsize = 20
    
    '''-------- Settings 1: schwach gedampft--------'''
    sets1 = general_sets.clone()
    sets1.dataset_file_name = "T0000.csv"
    
    '''-------- Settings 2: stark gedampft--------'''
    sets2 = general_sets.clone()
    sets2.dataset_file_name = "T0015.csv"

    '''-------- Settings 3: frei--------'''
    sets3 = general_sets.clone()
    #sets3.graph_format = "b."
    sets3.dataset_file_name = "T0029.csv"
    sets3.fitted_graph_label = "Gefittete Kurve: " + r"$y=A\:\cos(\omega t + \phi)$"

    '''=================================='''
    general_sets_xy = general_sets.clone()
    general_sets_xy.graph_format = ["b.","r-"] #first one for raw datapoints and second for smoothed ones
    general_sets_xy.graph_label = ["","Schwingungskurve im Phasenraum "]#["Datenpunte", "Gefilterte Datenpunkte"]
    general_sets_xy.x_label = "Auslenkung (V)"
    general_sets_xy.y_label = "Winkelgeschwindigkeit (V)"


    '''-------- Settings 4: schwach gedampft xy--------'''
    sets4 = general_sets_xy.clone()
    sets4.dataset_file_name = sets1.dataset_file_name
    sets4.graph_label[1] += "\nmit schwacher " + r"$D\"ampfung$"

    '''-------- Settings 5: stark gedampft xy--------'''
    sets5 = general_sets_xy.clone()
    sets5.dataset_file_name = sets2.dataset_file_name
    sets5.graph_label[1] += "\nmit starker " + r"$D\"ampfung$"
    sets5.y_label = "Phasenverschobene\nSinuschwingung\n(falsch aufgenommen)"

    '''-------- Settings 6: frei xy--------'''
    sets6 = general_sets_xy.clone()
    sets6.dataset_file_name = sets3.dataset_file_name
    sets6.graph_label[1] += "\n der freien Schwingung"

    return [sets1, sets2, sets3, sets4, sets5, sets6]

def smooth(data,filter_length):
    #smooths a curve by taking the averave value of each neighboring data point in filter_length range
    return [np.average(data[i-filter_length:i+filter_length]) for i in range(filter_length,len(data)-filter_length)]

def get_data(sets):
    #reads data from datasets specified in Settings
    dataset = dm.csv_to_list(sets.dataset_folder + sets.dataset_file_name)
    time_not_from_zero = dm.return_column (dataset, title = "TIME", title_index = sets.title_index)
    time = np.array(time_not_from_zero) - time_not_from_zero[0]
    deviation = dm.return_column (dataset, title = "CH2", title_index = sets.title_index)
    angular_velociy = dm.return_column (dataset, title = "CH1", title_index = sets.title_index)
    return time, deviation, angular_velociy

def plot_dampf_smooth(sets,filter_length):
    f = ft.exponential_decay_oscillator
    fit_samples_number = 500

    time, deviation_raw, a = get_data(sets)
    raw_params, raw_cov = curve_fit(f,time,deviation_raw)
    print ("This is raw covariance: " + str(raw_cov))
    #sets.fitted_graph_label += "\n" + r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.9f}\:$".format(raw_params[1],raw_params[2]) + r"$s^{-1}$"
    offset = raw_params[-1]
    deviation = [d-offset for d in deviation_raw]
    raw_params[-1] = 0
    
    deviation_smooth = smooth(deviation,filter_length)
    time_smooth = time[filter_length:-filter_length]
    smooth_params, smooth_cov = curve_fit(f,time_smooth,deviation_smooth)

    #old uncertainties
    #raw_freq, raw_beta = raw_params[1], raw_params[2]
    #raw_freq_str, raw_beta_str = error_string(raw_freq, np.sqrt(raw_cov[1][1])), error_string(raw_beta, np.sqrt(raw_cov[2][2]))
    #smooth_freq, smooth_beta = smooth_params[1], smooth_params[2]
    #smooth_freq_str, smooth_beta_str = error_string(smooth_freq, np.sqrt(smooth_cov[1][1])), error_string(smooth_beta, np.sqrt(smooth_cov[2][2]))
    
    #uncertainties
    fmtr = ShorthandFormatter()
    raw_freq_u, raw_beta_u = ufloat(raw_params[1], np.sqrt(raw_cov[1][1])), ufloat(raw_params[2], np.sqrt(raw_cov[2][2]))
    raw_freq_str, raw_beta_str = fmtr.format("{0:.1u}",raw_beta_u), fmtr.format("{0:.1u}", raw_freq_u)
    smooth_freq_u, smooth_beta_u = ufloat(smooth_params[1], np.sqrt(smooth_cov[1][1])), ufloat(smooth_params[2], np.sqrt(smooth_cov[2][2]))
    smooth_freq_str, smooth_beta_str = fmtr.format("{0:.1u}", smooth_beta_u), fmtr.format("{0:.1u}", smooth_freq_u)

    raw_k_u = exp(raw_beta_u/raw_freq_u)
    raw_k_str = fmtr.format("{0:.1u}",raw_k_u)
    smooth_k_u = exp(smooth_beta_u/smooth_freq_u)
    smooth_k_str = fmtr.format("{0:.1u}",smooth_k_u)


    x_fit = np.linspace(min(time), max(time), fit_samples_number)
    y_fit = [f(x,*raw_params) for x in x_fit]
    y_fit_smooth = f(x_fit, *smooth_params)
    print ("LENGTH OF YS: " + str(len(y_fit_smooth)) + " " + str(len(y_fit)))
    print ("This is smooth covariance: " + str(smooth_cov))
    #sets.graph_label = ["","",
    #                    r"$y = A\:\cos(\omega t + \phi)\: e^{-\beta t}$" + "\n"
    #                    r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.9f}\:$".format(raw_params[1],raw_params[2]) + r"$s^{-1}$",
    #                    r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.9f}\:$".format(smooth_params[1],smooth_params[2]) + r"$s^{-1}$"]
    
    sets.graph_label = ["","",
                        r"$y = A\:\cos(\omega t + \phi)\: e^{-\beta t}$" + "\n"
                        r"$\nu= \omega/2\pi = {} \:Hz,\ \beta = {}\:$".format(raw_freq_str,raw_beta_str) + r"$s^{-1}$" + "\nk = {}".format(raw_k_str),
                        r"$\nu= \omega/2\pi = {} \:Hz,\ \beta = {}\:$".format(smooth_freq_str,smooth_beta_str) + r"$s^{-1}$" + "\nk = {}".format(smooth_k_str)]
    
    fig, ax = plot_multi_2(sets = sets, x_values = [time,time_smooth,x_fit,x_fit], y_values = [deviation,deviation_smooth,y_fit,y_fit_smooth])

def plot_dampf(sets):
    f = ft.exponential_decay_oscillator

    sets.graph_format = sets.graph_format[0]

    time, deviation_raw, a = get_data(sets)
    raw_params, raw_cov = curve_fit(f,time,deviation_raw)
    #print ("This is raw covariance: " + str(raw_cov))
    #sets.fitted_graph_label += "\n" + r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.9f}\:$".format(raw_params[1],raw_params[2]) + r"$s^{-1}$"
    offset = raw_params[-1]
    deviation = [d-offset for d in deviation_raw]
    raw_params[-1] = 0

    #print (raw_params)
    #old uncertainties
    #freq, beta = raw_params[1], raw_params[2]
    #freq_str, beta_str = error_string(freq, np.sqrt(raw_cov[1][1])), error_string(beta, np.sqrt(raw_cov[2][2]))

    #uncertainties
    fmtr = ShorthandFormatter()
    A_u, freq_u = ufloat(raw_params[0], np.sqrt(raw_cov[0][0])), ufloat(raw_params[1], np.sqrt(raw_cov[1][1]))
    beta_u, phi_u = ufloat(raw_params[2], np.sqrt(raw_cov[2][2])), ufloat(raw_params[3], np.sqrt(raw_cov[3][3]))
    A_str, freq_str, omega_str = fmtr.format("{0:.1u}",A_u), fmtr.format("{0:.1u}", freq_u), fmtr.format("{0:.1u}", 2*np.pi*freq_u)
    beta_str, phi_str = fmtr.format("{0:.1u}",beta_u), fmtr.format("{0:.1u}", phi_u)

    print ("A = {}, BETA = {}, OMEGA = {}, PHI = {}".format(A_str, beta_str, omega_str, phi_str))
    
    k_u = exp(beta_u/freq_u)
    k_str = fmtr.format("{0:.1u}",k_u)

    #sets.fitted_graph_label += "\n" + r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.3f}\:$".format(raw_params[1],raw_params[2]) + r"$s^{-1}$"
    sets.fitted_graph_label += "\n" + r"$\nu= \omega/2\pi = {} \:Hz,\ \beta = {}\:$".format(freq_str,beta_str) + r"$s^{-1}$" + "\nk = {}".format(k_str)

    fig, ax = plot(sets, time, deviation, f_to_fit = f, params = raw_params)

def plot_frei_smooth(sets,filter_length):
    f = ft.cos_phase
    fit_samples_number = 500

    time, deviation, a = get_data(sets)
    raw_params, raw_cov = curve_fit(f,time,deviation, p0 = [2,0.8,0])
    #print ("This is raw covariance: " + str(raw_cov))
    #sets.fitted_graph_label += "\n" + r"$\nu= 2\pi\omega = {:.4f} \:Hz,\ \beta = {:.9f}\:$".format(raw_params[1],raw_params[2]) + r"$s^{-1}$"
    
    deviation_smooth = smooth(deviation,filter_length)
    time_smooth = time[filter_length:-filter_length]
    smooth_params, smooth_cov = curve_fit(f,time_smooth,deviation_smooth, p0 = [2,0.8,0])

    #old uncertainties
    #raw_freq = raw_params[1]
    #raw_freq_str = error_string(raw_freq, np.sqrt(raw_cov[1][1]))
    #smooth_freq = smooth_params[1]
    #smooth_freq_str = error_string(smooth_freq, np.sqrt(smooth_cov[1][1]))

    fmtr = ShorthandFormatter()
    raw_freq_u = ufloat(raw_params[1], np.sqrt(raw_cov[1][1]))
    raw_freq_str = fmtr.format("{0:.1u}", raw_freq_u)
    smooth_freq_u = ufloat(smooth_params[1], np.sqrt(smooth_cov[1][1]))
    smooth_freq_str = fmtr.format("{0:.1u}", smooth_freq_u)

    
    x_fit = np.linspace(min(time), max(time), fit_samples_number)
    y_fit = [f(x,*raw_params) for x in x_fit]
    y_fit_smooth = f(x_fit, *smooth_params)
    #print ("LENGTH OF YS: " + str(len(y_fit_smooth)) + " " + str(len(y_fit)))
    #print ("This is smooth covariance: " + str(smooth_cov))
    #sets.graph_label = ["","",
    #                    r"$y = A\:\cos(\omega t + \phi)$" + "\n"
    #                    r"$\nu= 2\pi\omega = {:.9f} \:Hz$".format(raw_params[1]),
    #                    r"$\nu= 2\pi\omega = {:.9f} \:Hz$".format(smooth_params[1])]
    sets.graph_label = ["","",
                        r"$y = A\:\cos(\omega t + \phi)$" + "\n"
                        r"$\nu= \omega/2\pi = {} \:Hz$".format(raw_freq_str),
                        r"$\nu= \omega/2\pi = {} \:Hz$".format(smooth_freq_str)]
    
    fig, ax = plot_multi_2(sets = sets, x_values = [time,time_smooth,x_fit,x_fit], y_values = [deviation,deviation_smooth,y_fit,y_fit_smooth])


def plot_frei(sets):
    f = ft.cos_phase
    time, deviation, a = get_data(sets)
    sets.graph_format = sets.graph_format[0]
    
    params, cov = curve_fit(f, time, deviation, p0 = [2,0.8,0])
    
    #
    #freq = params[1]
    #freq_str = error_string(freq, np.sqrt(cov[1][1]))

    fmtr = ShorthandFormatter()
    A_u, freq_u, phi_u = ufloat(params[0], np.sqrt(cov[0][0])), ufloat(params[1], np.sqrt(cov[1][1])), ufloat(params[2], np.sqrt(cov[2][2]))
    A_str, freq_str, omega_str, phi_str = fmtr.format("{0:.1u}", A_u), fmtr.format("{0:.1u}", freq_u),fmtr.format("{0:.1u}", 2*np.pi*freq_u), fmtr.format("{0:.1u}", phi_u)
    
    print ("A = {}, OMEGA = {}, PHI = {}".format(A_str, omega_str, phi_str))
    
    #print ("This is free covariance: " + str(cov))
    #sets.fitted_graph_label += "\n" + r"$\nu= 2\pi\omega = {:.4f} \:Hz$".format(params[1])
    sets.fitted_graph_label += "\n" + r"$\nu= \omega/2\pi = {} \:Hz$".format(freq_str)
    fig, ax = plot(sets, time, deviation, f_to_fit = f, params = params)


def plot_xy_smooth(sets, filter_length, plot_both = False):
    time, deviation_raw, angular_velocity_raw = get_data(sets)
    
    deviation_smooth = smooth(deviation_raw,filter_length)
    angular_velocity_smooth = smooth(angular_velocity_raw,filter_length)
    
    if plot_both:
        fig, ax = plot_multi_2(sets, [deviation_raw,deviation_smooth], [angular_velocity_raw,angular_velocity_smooth])
    else:
        sets.graph_format = sets.graph_format[1]
        sets.graph_label = sets.graph_label[1]
        fig, ax = plot(sets, deviation_smooth, angular_velocity_smooth)


def plot_xy(sets):
    time, deviation, angular_velocity = get_data(sets)
    print ("ANGULAR VEL DATA: " + str(len(angular_velocity)))
    sets.graph_format = sets.graph_format[0]
    length = len(time)
    fig, ax = plot(sets, deviation[:length], angular_velocity[:length])

'''============================================='''
if __name__ == "__main__":
    smoothen = True
    xy = True
    plot_raw_and_smooth = False
    filter_length = 20
    sets_list = get_settings()
    
    if (smoothen):
        if not xy:
            plot_dampf_smooth(sets_list[0],filter_length)
            plot_dampf_smooth(sets_list[1],filter_length)
            plot_frei_smooth(sets_list[2],filter_length)
        else:
            for sets in sets_list[3:]:
                plot_xy_smooth(sets, filter_length,plot_raw_and_smooth)
    else:
        if not xy:
            plot_dampf(sets_list[0])
            plot_dampf(sets_list[1])
            plot_frei(sets_list[2])
        else:
            for sets in sets_list[3:]:
                plot_xy(sets)
    plt.show()