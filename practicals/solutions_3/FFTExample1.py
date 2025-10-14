from scipy.fftpack import fft, fftfreq
from numpy import linspace, sin, pi

#Define the length of our box
length = 1.0
nx = 100
x = linspace(0,length,nx,endpoint=False)
dx = x[1]-x[0]

#Make our function, sum two sin curves; f = sin(x k_25)+sin(x*k_10)/2
#where k_i is the ith wave-number = 2*pi/lambda_i
f = sin(x*2*pi*25)+sin(x*2*pi*10)*0.5

#FFT; Converts from x to k_x
fHat = fft(f)

#Now calculate the x values of the transformed problem, i.e. the inverse 
#wavelengths lambda. 
#Note we, only use the first half of the solution because of the fft
#symmetry (+ve and -ve wavelengths) -- See the documentation
lamb = fftfreq(nx,dx)
upLim = nx//2
#Note fftfreq is doing something like:
#lamb = linspace(1.0,1.0/(2*dx),nx/2,endpoint=False)

#Plot spectrum
import matplotlib.pyplot as plt
plt.plot(lamb[:upLim],(2.0/nx)*abs(fHat[:upLim]),'-x')
plt.xlabel(r'$1/\lambda$') ; plt.ylabel(r'$\hat{f}$')
plt.grid() ; plt.show()
