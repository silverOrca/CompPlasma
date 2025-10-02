# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 15:29:31 2025

@author: zks524

Code for variable in dy/dt - not hardcoded
"""


from scipy.integrate import solve_ivp
from numpy import linspace 
import matplotlib.pyplot as plt

# Here we define our function which returns df/dt = -a*f (the RHS)
#Takes parameters of current time, current function value and 'a' value
def dfdt (curT, curF, aVar):
    #We don ’t do anything with curT
    return -aVar * curF


#Now define the times at which we want to know the result
#This says the points at which the steps should be taken, the points which 
#are shown in the output
time = linspace (0 ,1 ,40)

#Set the initial condition - the array of initial conditions passed into solve_ivp
f0 = [10.]


#Define array of various variable values
varA = [0, 1, 10, 30, -1]

#loop through array of 'a' variables and calulate result for each function
for a in varA:
    #Now we can solve the equation
    result = solve_ivp (dfdt, [ time [0] , time [ -1]], f0, t_eval = time, args=(a,))

    #plot each one
    plt.plot(time, result.y[0,:], label = 'a = '+str(a))

# Plot

#Plots numerical solution, calculated using solve_ivp
#plt.plot ( time, result.y[0,:], 'x', label = 'odeint')

#Plots analytic solution calculated in the brackets
#plt.plot ( time, f0 * exp ( -10.* time ), label = 'analytic')

plt.xlabel ('Time'); plt . ylabel ("f")
plt.legend (loc='best') ; plt.show ()
