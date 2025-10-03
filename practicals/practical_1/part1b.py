# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 15:37:49 2025

@author: zks524

Simple integrate with no hardcoded (but default value) variables and sin cos function
"""

from scipy.integrate import solve_ivp
from numpy import linspace, sin, cos
import matplotlib.pyplot as plt

def dfdt(curT, curF, aVar=30, bVar=50):
    return (-aVar * sin(curF)) - (bVar * cos(curT))


time = linspace(0, 10, 400)

f0=[10.]

result = solve_ivp(dfdt, [time[0], time[-1]], f0, t_eval=time)

plt.plot(time, result.y[0,:], label = 'a function!')

plt.xlabel ('Time'); plt.ylabel ("f")
plt.legend (); plt.show ()

