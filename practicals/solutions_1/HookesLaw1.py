#Import useful routines from modules
from scipy.integrate import solve_ivp
from numpy import linspace, exp
import matplotlib.pyplot as plt

#Here we define our function which returns d[x,v]/dt = [v,-kx/m]
def dfdt(curT,curF,springK=1,mass=1,damping=0.01):
    #Unpack values
    x = curF[0] ; v = curF[1]
    #Calculate time derivative of two terms
    dxdt = v  #dx/dt=v
    dvdt = -(springK*x/mass)-(damping*v/mass) # dv/dt = -(k*x/m)-(c*v/m)
    return [dxdt,dvdt]
 
#Now define the times at which we want to know the result
time=linspace(0,10,400)

#Set the initial conditions
x0=0.4 ; v0=0. ; f0=[x0,v0]

#Set the spring and mass parameters
mass = 1.0e-2 ; k = 0.5 ; damping = 0.005

#Now we can solve the equation
result = solve_ivp(dfdt,[time[0], time[-1]], f0,
                t_eval = time, args=(k,mass,damping))

#Extract x and v
x_of_t = result.y[0,:] ; v_of_t = result.y[1,:]

##Plot
plt.plot(time,x_of_t,label='x(t)') ; plt.plot(time,v_of_t,label='v(t)')
plt.xlabel('Time'); plt.ylabel("x and v") ; plt.legend(loc='best') ; plt.show()

#Energy plot -- Uncomment the below to produce plot (comment out above plots)
##Define a function that returns the kinetic, potential and total energy
#def getEnergy(x,v,k,mass):
#    kE=0.5*mass*v*v ; pE=0.5*k*x*x ; tE=kE+pE
#    return kE,pE,tE

#ke,pe,te=getEnergy(x_of_t,v_of_t,k,mass)
#plt.plot(time,ke,label="Kinetic energy")
#plt.plot(time,pe,label="Potential energy")
#plt.plot(time,te,label="Total energy")
#plt.xlabel('Time'); plt.ylabel("Energy")
#plt.legend(loc='best') ; plt.show()
