# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 10:16:51 2025

@author: ciara

Hooke's law code
"""

#need to define the functions of what dv/dt are, as well as what we are defining v as
#here, dfdt is the d^x/dt^2


from scipy.integrate import solve_ivp
from numpy import linspace, exp
import matplotlib.pyplot as plt

#define Hooke's law function
def dfdt(curT, curF, m, k):
    #need to figure out what the current x and v values are
    #x should be the very first value of our function f0, and v is the second one
    x = curF[0]
    v = curF[1]
    
    #define what v is, so we can have that changing with time first (solve_ivp) to plot
    dxdt = v
    
    #Hooke's law
    dvdt = (-k * x / m) - (c * v / m)
    
    return [dxdt, dvdt]

m=0.01; k=0.5

time = linspace(0, 20, 400)

#define initial values for position and velocity
x0=0.1
v0=0.

#damping value
c=0.005

#they need to be passed in as part of a function, which we will pass in dfdt as curF
f0=[x0, v0]

result = solve_ivp(dfdt, [time[0], time[-1]], f0, t_eval=time, args=(m, k))

#separates the results into individual sets of values so we can plot each
xResult = result.y[0,:]
vResult = result.y[1,:]

#plot x values
plt.plot(time, xResult, label = "x(t)")
#plot v values
plt.plot(time, vResult, label ='v(t)')

plt.xlabel('Time'); plt.ylabel('Position and Velocity')
plt.legend(loc = 'best'); plt.show()


#Finding the potential, kinetic and total energy of the spring
def energy(x, k, m, v):
    #potential energy equation
    u = 0.5 * k * (x**2)
    #kinetic energy equation
    e_k = 0.5 * m * (v**2)
    #total energy equation
    e_tot = u + e_k
    
    return u, e_k, e_tot
  
u, e_k, e_tot = energy(xResult, k, m, vResult)
      
plt.plot(time, u, label='Potential energy')
plt.plot(time, e_k, label='Kinetic energy')
plt.plot(time, e_tot, label='Total energy')

plt.xlabel('Time'); plt.ylabel('Energy')
plt.legend(loc='best'); plt.show()





