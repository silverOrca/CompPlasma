# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 09:38:54 2025

@author: zks524

Lecture 2

Code for solving dy/dt = 10 * y explicitly

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

#Work out time element - how far we are going in time / steps
#more steps means time element is smaller, more accurate results
dt = (t_final - t0) / nsteps

#want to store the solution at each step - list. We have initial
#values so can put y0 and t0 in
solution = [y0]
time = [t0]

#this is the true solution to compare to our computational solution
analytic_solution = [y0 * exp(a_coefficient * time[0])]

#loop over the steps to calculate values at each dt
for i in range(nsteps):
    #go to last value in list to calculate from that the value that should be next
    y_current = solution[-1]
    #next value is last thing in list + next step amount
    #taken step of dt and multiplying by rate, linear extrapolation
    y_next = y_current + (dt * a_coefficient * y_current) #dy/dt = a_coefficient * y
    
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


#instead, could define step size and while loop until the time reaches the max time
#where each iteration increments the time by one step size

