def getForce(x,y):
    #Create array for force values
    N=len(x) ; Fx = zeros(N) ; Fy = zeros(N)
    #Now calculate the force on each particle
    for i in range(N):
        #Loop through all the other particles
        #to calculate their force on this particle
        for j in range(N):
            if i==j: continue #No force on self
            #Calculate distance
            dx = x[i] - x[j] ; dy = y[i] - y[j]
            R=sqrt(dx**2+dy**2)
            #Calculate force
            Fx[i] -= dx/R**3 ; Fy[i] -= dy/R**3
    return Fx, Fy

def f(time, state):
    #Work out N from state
    N = len(state) // 4

    #Unpack variables
    x = state[0:N] ; y = state[N:(2*N)]
    vx = state[(2*N):3*N] ; vy = state[(3*N):]

    #Find out the force
    Fx, Fy = getForce(x,y)

    #Concatenate takes list of B, N length vectors
    #and returns a single B*N vector
    #Note dx/dt=vx, dy/dt=vy, dvx/dt=Fx and dvy/dt=Fy
    return concatenate([ vx, vy, Fx, Fy ])


from numpy import zeros, sqrt, concatenate, linspace
from scipy.integrate import solve_ivp
#Initial conditions: 
#Pcle 1 at (x,y)=(-1,0) with (vx,vy)=(0,0.2)
#Pcle 2 at (x,y)=(1,0) with (vx,vy)=(0,-0.2)
initial = [-1.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.2,-0.2] 
#Times of interest
t = linspace(0, 20, 200)
#Integrate
result = solve_ivp(f, [t[0], t[-1]], initial, t_eval = t)
#Plot
import matplotlib.pyplot as plt
plt.plot( result.y[0,:], result.y[2,:])
plt.plot( result.y[1,:], result.y[3,:])
plt.show()

# Note how the orbits actually seem to decay a little.
# Try changing the integration method, how does it change
# with LSODA and DOP853 methods?
