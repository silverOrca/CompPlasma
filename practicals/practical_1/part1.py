# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 15:10:56 2025

@author: zks524

Code for hardcoded value of dy/dt function where dy/dt = -ay
"""

from scipy.integrate import solve_ivp
from numpy import linspace , exp
import matplotlib.pyplot as plt

# Here we define our function which returns df/dt = -a*f (the RHS)
# Note we ’ve assumed that a=10
#Takes parameters of current time and current function value
def dfdt ( curT, curF ):
    #We don ’t do anything with curT
    return -10* curF


#Now define the times at which we want to know the result
#This says the points at which the steps should be taken, the points which 
#are shown in the output
time = linspace (0 ,1 ,40)

#Set the initial condition - the array of initial conditions passed into solve_ivp
f0 = [10.]

#Now we can solve the equation
result = solve_ivp (dfdt ,[ time [0] , time [ -1]] , f0 , t_eval = time)

# Plot

#Plots numerical solution, calculated using solve_ivp
plt.plot ( time, result.y[0,:], 'x', label = 'odeint')

#Plots analytic solution calculated in the brackets
plt.plot ( time, f0 * exp ( -10.* time ), label = 'analytic')

plt.xlabel ('Time'); plt . ylabel ("f")
plt.legend () ; plt . show ()
