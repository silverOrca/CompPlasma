#Import useful routines from modules
from scipy.integrate import solve_ivp
from numpy import linspace, exp
import matplotlib.pyplot as plt

#Here we define our function which returns df/dt = -a*f
#Note we can pass in a, but it defaults to 10
def dfdt(curT,curF,a=10):
    #We don't do anything with curT
    return -a*curF

#Now define the times at which we want to know the result
time=linspace(0,1,40)

#Set the initial condition
f0=[10.]

#Which a values do we want to use?
avals=[0,1,10,30,-1]

for a in avals:
    #Now we can solve the equation for this a
    #Note we need a comma after the a here to
    #make sure args is a *tuple*
    result = solve_ivp(dfdt,[time[0], time[-1]], f0,
                       t_eval = time, args=(a,))

    plt.plot(time,result.y[0,:],label='a = '+str(a))

plt.xlabel('Time'); plt.ylabel("f")
plt.legend(loc='best') ; plt.show()
