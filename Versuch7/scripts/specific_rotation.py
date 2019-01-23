import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,"../../scripts")
import fit_functions as ft
from myPlot import ShorthandFormatter

lambda_literature = [656,589,509,436,373]
specific_rotation_literature = [17.3, 21.7, 29.7, 41.5, 58.9]
#lambda_literature = [656,589,509]#,436,373]
#specific_rotation_literature = [17.3, 21.7, 29.7]#, 41.5, 58.9]
lambda_experiment = [690,650,590,530]
specific_rotation_experiment = [18, 19, 22, 25]
specific_rotation_experiment_err = [1 for i in range(len(specific_rotation_experiment))]

f_l = ft.quartic
f_e = ft.linear2
popt_literature, cov_literature = curve_fit(f_l, lambda_literature, specific_rotation_literature)
popt_experiment, cov_experiment = curve_fit(f_e, lambda_experiment, specific_rotation_experiment, sigma = specific_rotation_experiment_err, absolute_sigma = True)

params_l = unp.uarray(popt_literature, np.sqrt(np.array([cov_literature[i][i] for i in range(len(cov_literature))])))
params_e = unp.uarray(popt_experiment, np.sqrt(np.array([cov_experiment[i][i] for i in range(len(cov_experiment))])))
fmtr = ShorthandFormatter()
params_l_str = [fmtr.format('{0:.1u}', params_l[i]) for i in range(len(params_l))]
params_e_str = [fmtr.format('{0:.1u}', params_e[i]) for i in range(len(params_e))]
x = np.linspace(350, 750, 1000)

fig, ax = plt.subplots()

ax.plot(x, f_l(x, *popt_literature), 'r-', label = 'fit on literature values')
#ax.plot(x, f_l(x, *popt_literature), 'y-', label = r'literature fit: $A={}\, \frac{{deg}}{{\mu m^2}}$, $b={}\, \frac{{deg}}{{mm}}$'.format(params_l_str[0], params_l_str[1]))
#ax.plot(x, f_e(x, *popt_experiment), 'r-', label = r'experimental fit: $A={}\, \frac{{deg}}{{\mu m^2}}$, $b={}\, \frac{{deg}}{{mm}}$'.format(params_e_str[0], params_e_str[1]))
ax.plot(lambda_literature, specific_rotation_literature, 'go', label = 'Literature Values')
ax.errorbar(lambda_experiment, specific_rotation_experiment, yerr = specific_rotation_experiment_err, fmt = 'bo', label = 'Experimental values', capsize = 2)

ax.set_xlabel('Wavelength in nm', fontsize = 18)
ax.set_ylabel('Specific rotation in deg/mm', fontsize = 18)
#ax.set_xbound(490, 710)
#ax.set_ybound(11, 49)
ax.set_xbound(350, 710)
ax.set_ybound(14, 71)

legend = ax.legend(fontsize = 18)#, title = r'Linear fit: $y=A\cdot x + b$')
#legend.get_title().set_fontsize('14')

ax.tick_params(labelsize = 18)
plt.tight_layout()
plt.subplots_adjust(left = 0.12, right = 0.99, bottom = 0.135, top = 0.99)

plt.show()