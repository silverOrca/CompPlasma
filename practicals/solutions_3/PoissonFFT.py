#Program to solve 1D Poisson using FFTs
from scipy.fftpack import fft, fftfreq, ifft
import matplotlib.pyplot as plt
from numpy import linspace, sin, pi, sqrt, mean, zeros

#Define the length of our box
length = 1.0
nx = 1000
x = linspace(0,length,nx,endpoint=False)
dx = x[1]-x[0]

#Make our rhs function, rho = sin(2 Pi (x/length))
rho = sin((x/length)*2*pi)

#Now FFT to get rhoHat
rhoHat = fft(rho)
#Get the inverse wavelengths
lambInv = fftfreq(nx,dx)
#Calculate wavenumbers
kx = lambInv*2*pi

#Now we can find phiHat using phiHat = rhoHat / kx^2 epsilon0
#Note: we've normalised such that epsilon0==1.0
phiHat = zeros(len(rhoHat),dtype='complex')#Have to make this complex!
#Note we skip the kx=0 component as this just defines a DC offset
#which we'll assume is zero
phiHat[1:] = rhoHat[1:] / kx[1:]**2
#Now we can find phi
phi = ifft(phiHat)


#Analytic solution
phiAn = sin(2*pi*x)/(2*pi)**2
err = phiAn-phi
rmsErr = sqrt(mean(err*err.conjugate()))
print("The rms error is {err}".format(err=rmsErr))

#Now plot
plt.plot(x,phi.real,'-',label=r'$\phi$ Numeric')
plt.plot(x[::10],phiAn[::10],'x',label=r'$\phi$ Analytic')
plt.xlabel(r'$x$') ; plt.ylabel(r'$\phi$')
plt.legend(loc='best'); plt.show()
