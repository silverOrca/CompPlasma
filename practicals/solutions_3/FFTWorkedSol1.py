from scipy.fftpack import fft, fftfreq, fftshift, ifftshift, ifft
import matplotlib.pyplot as plt
from numpy import linspace, sin, pi, sqrt, mean

#Define the length of our box
length = 1.0
nx = 1000
x = linspace(0,length,nx,endpoint=False)
dx = x[1]-x[0]

#Make our function, sum two sin curves; f = sin(x k_25)+sin(x*k_10)/2
#where k_i is the ith wave-number = 2*pi/lambda_i
f = sin(x*2*pi*25)+sin(x*2*pi*10)*0.5

#FFT; Converts from x to k_x
fHat = fft(f)
#Calculate the inverse wavelengths
lamb = fftfreq(nx,dx)
#Shift to put -ve values first then positive
fHat = fftshift(fHat)
lamb = fftshift(lamb)
#Plot full spectrum
#plt.plot(lamb,(1.0/nx)*abs(fHat),'-x')
#plt.xlabel(r'$1/\lambda$') ; plt.ylabel(r'$\hat{f}$')
#plt.grid() ; plt.show()

#/Now shift order back
fHat = ifftshift(fHat)
lamb = ifftshift(lamb)

#Let's calculate the rms error on the back transform
fBackTrans = ifft(fHat) ; err = f-fBackTrans
rmsErr = sqrt(mean(err*err.conjugate()))
print("The rms error is {err}".format(err=rmsErr))

#Now let's filter out the lambda = 1.0/10.0 wave
#Find the index where lamb is closest to 10.0
indPos = abs(10.0-lamb).argmin()
#Find the index where lamb is closest to -10.0
indNeg = abs(10.0+lamb).argmin()
#Zero out the coefficients at these locations
fHat[[indPos,indNeg]] = 0.0
#Now we can inverse transform
fFilt = ifft(fHat)
#Plot
plt.plot(x,f,label='Original function')
plt.plot(x,fFilt,label='Filtered function')
plt.xlabel(r'$x$') ; plt.ylabel(r'$f$')
plt.legend(loc='best') ; plt.show()
