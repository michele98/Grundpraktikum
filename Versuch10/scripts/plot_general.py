import matplotlib.pyplot as plt
import data_manager as dm
import math
import fit_functions as ft
from scipy.optimize import curve_fit


def frange(start, end=None, inc=None):
    """A range function, that does accept float increments..."""
    import math

    if end == None:
        end = start + 0.0
        start = 0.0
    else: start += 0.0 # force it to be a float

    if inc == None:
        inc = 1.0
    count = int(math.ceil((end - start) / inc))

    L = [None,] * count

    L[0] = start
    for i in xrange(1,count):
        L[i] = L[i-1] + inc
    return L

def plot (x_values, y_values, x_label, y_label, x_err = 0, y_err = 0, f = ft.linear):
    popt, pcov = curve_fit(f,x_values, y_values)

    plt.errorbar(x_values,y_values, xerr=x_err, yerr=y_err, fmt = "bs")
    #plt.plot(x_values, y_values, "b.")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    x_fit = [i for i in frange(x_values[0], x_values[-1])]
    y_fit = [f(i,*popt) for i in frange(x_values[0], x_values[-1])]
    plt.plot (x_fit, y_fit, "r-")
    
    plt.show()

if __name__ == "__main__":
    #datsets csv files have to be put in datasets folder
    #file name has to be written without extension
    file_folder = "../datasets/"

    '''
    #dirty way to have it work both python3 and python2
    try:
        dataset_name = input("Dataset name: ")
        x_caption = input("x Axis Data: ")
        y_caption = input("y Axis Data: ")
        x_err_caption = input("x Error: ")
        y_err_caption = input("y Error: ")
        y_caption = input("y Axis Label: ")
        y_caption = input("y Axis Label: ")
        include_zero = input ("include zero?: ")

    except:
        dataset_name = raw_input("Dataset name: ")
        x_caption = raw_input("x Axis Label: ")
        y_caption = raw_input("y Axis Label: ")
        x_err_caption = raw_input("x Error: ")
        y_err_caption = raw_input("y Error: ")
        y_caption = input("y Axis Label: ")
        y_caption = input("y Axis Label: ")
        include_zero = raw_input ("include zero?: ")
    '''

    dataset_name = "strsp1"
    x_caption = "Ur"
    y_caption = "R"
    x_err_caption = "fUr"
    y_err_caption = "fR"
    x_label = "Spannung an der Lampe Ur [V]"
    y_label = "Lamperwirderstand R [Ohm]"
    include_zero = "no"
    fit_f = ft.cubic
    
    dataset = dm.csv_to_list(file_folder + dataset_name + ".csv")
    x_values = dm.return_column (dataset, name = x_caption)
    x_err_values = dm.return_column (dataset, name = x_err_caption)
    y_values = dm.return_column (dataset, name = y_caption)
    y_err_values = dm.return_column (dataset, name = y_err_caption)

    if (include_zero == "yes"):
        x_values.append(0.)
        x_err_values.append(0.)
        y_values.append(0.)
        y_err_values.append(0.)
    
    print (x_values)
    print (y_values)
    plot(x_values, y_values, x_label, y_label, x_err_values, y_err_values, fit_f)