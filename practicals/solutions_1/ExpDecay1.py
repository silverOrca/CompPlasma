#Import useful routines from modules
from scipy.integrate import solve_ivp
from numpy import linspace, exp
import matplotlib.pyplot as plt

#Here we define our function which returns df/dt = -a*f
#Note we've assumed that a=10
def dfdt(curT,curF):
    #We don't do anything with curT
    return -10*curF

#Now define the times at which we want to know the result
time = linspace(0,1,40)

#Set the initial condition
f0 = [10.]

#Now we can solve the equation
result = solve_ivp(dfdt,[time[0], time[-1]], f0, t_eval = time)

#Plot
plt.plot(time, result.y[0,:],'x', label='odeint')
plt.plot(time, f0*exp(-10.*time), label='analytic')
plt.xlabel('Time'); plt.ylabel("f")
plt.legend() ; plt.show()
