#Import useful routines from modules
from scipy.integrate import solve_ivp
from numpy import linspace, sin
import matplotlib.pyplot as plt

#Here we define our function which returns d[theta,omega]/dt = [omega,-g theta/l]
def dfdt(curT,curF,length=1.0,g=9.81,useSmall=False):
    #Unpack values
    theta = curF[0] ; omega = curF[1]

    #Calculate time derivative of two terms
    dthetadt = omega 
    if useSmall:
        domegadt = -g*theta/length
    else:
        domegadt = -g*sin(theta)/length
    return [dthetadt,domegadt]
 
#Now define the times at which we want to know the result
time=linspace(0,10,40)

#Set the initial conditions, note theta is in radians
theta0=0.5 ; omega0=0. ; f0=[theta0,omega0]

#Set the parameters
g = 9.81 ; length = 2

#Now we can solve the equation
result = solve_ivp(dfdt, [time[0], time[-1]], f0,
                   t_eval = time, args=(length,g))
plt.plot(time,result.y[0,:],label=r'$\theta(t)$ -- Full')
plt.plot(time,result.y[1,:],label=r'$\omega(t)$ -- Full')

result = solve_ivp(dfdt, [time[0], time[-1]], f0,
                   t_eval = time, args=(length,g,True))
plt.plot(time,result.y[0,:],'--x',label=r'$\theta(t)$ -- Small angle')
plt.plot(time,result.y[1,:],'--x',label=r'$\omega(t)$ -- Small angle')
plt.xlabel('Time'); plt.ylabel(r"$\theta$ and $\omega$")
plt.legend(loc='best') ; plt.show()
