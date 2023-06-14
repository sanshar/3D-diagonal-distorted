import numpy as np
import scipy.io
import matplotlib.pyplot as plt


mat = scipy.io.loadmat('gauss_coeff.mat')
Coeff = mat['coeff'][:,0]
sigma = 2./3.
xi = np.arange(-50,50.22,1./3.)

def getVal(x):
    if isinstance(x, np.ndarray):
        out = 0.*x
        for i in range(x.shape[0]):
            out[i] = np.sum(Coeff * np.exp(-((x[i]-xi)/sigma)**2/2. ))
        return out
    else:
        return np.sum(Coeff * np.exp(-((x-xi)/sigma)**2/2. ))
    
##
def KE():
    Ngrid = 1000
    grid = np.linspace(-100,100,Ngrid+1)[:-1] ##pick the grid
    G = np.fft.fftfreq(Ngrid) * 2 * np.pi * Ngrid / 200.    
    dx = grid[1] - grid[0]
    
    f1 = getVal(grid)
    
    KE = np.zeros((45,))
    for i in range(-22,22):
        f2 = getVal(grid-i)
        f2g = np.fft.ifft(f2)
        f2g = G**2/2 * f2g

        f2 = np.fft.fft(f2g)
        KE[i+22] = np.dot(f1,f2)*dx       
        print(i, KE[i+22].real)
        
    return np.asarray(KE)