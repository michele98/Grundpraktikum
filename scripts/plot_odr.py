from scipy.odr import ODR, Model, Data, RealData
import numpy as np
import matplotlib.pyplot as plt
import fit_functions_odr as ft

def func(beta, x):
    y = beta[0]+beta[1]*x+beta[2]*x**3
    return y

#generate data
x = np.linspace(-3,2,100)
y = func([-2.3,7.0,-4.0], x)

# add some noise
x += np.random.normal(scale=0.3, size=100)
y += np.random.normal(scale=0.1, size=100)

data = RealData(x, y, 0.3, 0.1)
model = Model(ft.exp1)

odr = ODR(data, model, [1,1])
odr.set_job(fit_type=2)
output = odr.run()

xn = np.linspace(-3,2,50)
yn = func(output.beta, xn)
plt.plot(x,y,'ro')
plt.plot(xn,yn,'k-',label='leastsq')
odr.set_job(fit_type=0)
output = odr.run()
yn = func(output.beta, xn)
plt.plot(xn,yn,'g-',label='odr')
plt.show()