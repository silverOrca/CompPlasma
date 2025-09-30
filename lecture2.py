# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 09:38:54 2025

@author: zks524

Lecture 2

Code for solving dy/dt = 10 * y

y(t) = y(t=0) * exp(a*t)
"""

import matplotlib.pyplot as plt
from math import exp

#need initial value for y
y0 = 1.0

#need value for the variable
a_coefficient = 10

#need initial time, useful for evalutating true solution
t0 = 0 

#final time
t_final = 1.0

#how many steps do we want to take in the integration?
nsteps = 101


#Implement the method

#Work out time step - how far we are going in time / steps
dt = (t_final - t0) / nsteps

#want to store the solution at each step - list. We have initial
#values so can put y0 and t0 in
solution = [y0]
time = [t0]

analytic_solution = [y0 * exp(a_coefficient *time[0])]

#loop over the steps
for i in range(nsteps):
    y_current = solution[-1]
    #next value is last thing in list +
    #taken step of dt and multiplying by rate, linear extrapolation
    y_next = y_current + dt * a_coefficient * y_current #dy/dt = a_coefficientefficient * y
    
    #use append to add new values to lists
    solution.append(y_next)
    time.append(time[-1]+dt)
    analytic_solution.append(y0 * exp(a_coefficient * time[-1]))
    

plt.plot(time, solution, '-0', label='Numerical')

#for this to work either need numpy array or new anayltic solution
plt.plot(time, analytic_solution, '-', label = 'Analytic')

plt.xlabel('Time')
plt.ylabel('State: y')
plt.legend(loc = 'best')

plt.show()





