#
# Fitting Chebyshev polynomials to data
# 

from numpy.polynomial.chebyshev import chebfit, chebval
from numpy import linspace, exp, tanh, pi, cos
import matplotlib.pyplot as plt

def func(x):
    return (x-0.5)*(0.2 + tanh(3.*x))


xall = linspace(-1.,1., 100, endpoint=True) # High resolution mesh for plotting
yall = func(xall)

#############################
# Equally spaced points
plt.subplot(221)
plt.plot(xall, yall)

n = 8
x = linspace(-1.,1.,n, endpoint=True)
y = func(x)

plt.plot(x, y, 'o')

# Fit Chebyshev polynomials
c = chebfit(x, y, n)

# Plot fit
plt.plot(xall, chebval(xall, c))
plt.title("Equally spaced points")
plt.ylabel(str(n)+" grid points")

############################
# Chebyshev points
plt.subplot(222)
plt.plot(xall, yall)
x = -cos(linspace(0, pi, n, endpoint=True))
y = func(x)
plt.plot(x, y, 'o')

c = chebfit(x, y, n)
plt.plot(xall, chebval(xall, c))
plt.title("Chebyshev points")

#############################
# Equally spaced points
plt.subplot(223)
plt.plot(xall, yall)
n = 20
x = linspace(-1.,1.,n, endpoint=True)
y = func(x)
plt.plot(x, y, 'o')

c = chebfit(x, y, n)
plt.plot(xall, chebval(xall, c))
plt.ylabel(str(n)+" grid points")

############################
# Chebyshev points
plt.subplot(224)
plt.plot(xall, yall)
x = -cos(linspace(0, pi, n, endpoint=True))
y = func(x)
plt.plot(x, y, 'o')

c = chebfit(x, y, n)
plt.plot(xall, chebval(xall, c))

plt.show()
