#Look at FFT coefficients
rhoHat = (1.0/nx)*(fft(rho,axis=1).imag)
invLamb = fftfreq(nx,dx)
plt.figure(2)
plt.contourf(fftshift(invLamb),t,fftshift(rhoHat,axes=1),64)
plt.xlabel(r'$1/\lambda$') ; plt.ylabel(r'$t$')
plt.colorbar(label=r'$\hat{\rho}$')
plt.show()