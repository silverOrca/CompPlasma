# Solve dy / dt = a * y
# Analytic y(t) = y(t=0) * exp(a * t)
# Use Euler's explicit method
# Initial conditions, y(t=0) = 1, a = 1
# Find y for 0<=t<=3
#Storage for time evolving data
times = []
result = []
analytic = []
#Initial conditions/parameters
y0 = 1.0
a = -10.0
t0 = 0.0
# Import exp
from math import exp
times.append(t0)
result.append(y0)
analytic.append(y0 * exp(a * t0))
# How far do we integrate
endTime = 1.0
time = t0
timeStep = 0.01
useImplicit = True

while time < endTime:
    curState = result[-1]
    
    if not useImplicit:
        newState = curState + timeStep * (a * curState )
        
    else:
        newState = curState / (1 - timeStep * a)
        
    time = time + timeStep
    
    result.append(newState)
    times.append(time)
    analytic.append(y0 * exp(a * time))

import matplotlib.pyplot as plt
plt.plot(times, result, '-o', label='Numerical solution')
plt.plot(times, analytic, '-', label = 'Analytic solution')
plt.legend(loc='best') ; plt.xlabel('Time') ; plt.ylabel('y')
print("The difference is {d}".format(
    d = analytic[-1] - result[-1]))
plt.grid(True,'both') ; plt.show()
