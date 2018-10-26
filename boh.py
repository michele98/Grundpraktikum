import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import numpy as np

def f(x,a,b):
    return a*x+b

xp = np.linspace(0,100)
yp = np.linspace(0,100)
xn = np.random.normal(size = xp.size)
yn = 1*np.random.normal(size = yp.size)
xv = xp+xn
yv = yp+yn
para, cov = curve_fit(f,xv,yv)
print (para)
print (cov)

plt.plot(xv, yv, "bo")
plt.plot([x for x in range(100)],[f(x,*para) for x in range(100)], "r-")
plt.show()