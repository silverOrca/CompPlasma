#Import useful routines from modules
from scipy.integrate import solve_ivp
from numpy import linspace, sin, cos
import matplotlib.pyplot as plt
import matplotlib.animation as ani

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
time=linspace(0,10,400)

#Set the initial conditions
theta0=3.5; omega0=0.; f0=[theta0,omega0]

#Set the parameters
g = 9.81 ; length = 2

#Now we can solve the equation
result = solve_ivp(dfdt, [time[0], time[-1]], f0,
                   t_eval = time, args=(length,g))

#Plot
fig1 = plt.figure(); plt.xlabel(r"$x$"); plt.ylabel(r"$y$")
line1 = plt.plot([],[],'-') #The line connecting origin and bob
line2 = plt.plot([],[],'go') #Pendulum bob

#Calculate positions and set the plot range
plt.xlim(-length,length); plt.ylim(-length,length)
fig1.gca().set_aspect('equal')

#Function to get pendulum x-y position given theta and length
def pendPos(theta,length):
    x = length*sin(theta) ; y = length*cos(theta)
    return x,y
#Function to draw the pendulum
def drawPendulum(num,res,l1,l2,length=1):
    x,y = pendPos(res[0,num],-length)
    l1.set_data([0,x],[0,y]) ; l2.set_data(x,y)
    return l1,l2,
#Make animation and then show it
line_ani = ani.FuncAnimation(fig1,drawPendulum,len(time),
    fargs=(result.y,line1[0],line2[0],length),
    interval=50, blit=False, repeat=False)
plt.show()
