def lv(time, state, alpha, beta, gamma, delta):
    # inputs are system state and
    # the simulation time   
    x = state[0] ; y = state[1]
    dxdt = alpha * x - beta*x*y
    dydt = -gamma*y + delta*x*y
    return [dxdt, dydt]

from scipy.integrate import solve_ivp
from numpy import linspace
initial = [5, 5]
t = linspace(0.0, 20, 200)
result = solve_ivp(lv, [t[0], t[-1]], initial,
                   t_eval = t, args=(1.0, 1.0, 3.0, 1.0))

import matplotlib.pyplot as plt
plt.plot(t, result.y[0,:], label="prey")
plt.plot(t, result.y[1,:], label="predator")
plt.legend() ; plt.show()
