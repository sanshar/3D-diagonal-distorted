import numpy as np

def PadZeroForInterpolation1d(freqVal, nmesh, nDenseMesh):
    freqVal = np.roll(freqVal, nmesh[0]//2 )
    freqValDense = np.zeros(nDenseMesh, dtype=complex)
    freqValDense[0:nmesh[0]] = freqVal

    '''
    if nmesh[0]%2 == 0:
        freqValDense[0] /= 2.
        freqValDense[nmesh[0]] = freqValDense[0]
    '''
    freqValDense = np.roll(freqValDense, -nmesh[0]//2)

    return freqValDense

def PadZeroForInterpolation2d(freqVal, nmesh, nDenseMesh):
    temp = np.roll(freqVal, ((nmesh[0]+1)//2, (nmesh[1]+1)//2), axis=(0,1) )

    freqValDense = np.zeros(nDenseMesh, dtype=complex)
    freqValDense[0:nmesh[0],0:nmesh[1]] = 1*temp

    '''
    if nmesh[0]%2 == 0:
        freqValDense[0,:] /= 2.
        freqValDense[nmesh[0],:] = freqValDense[0,:]
    if nmesh[1]%2 == 0:
        freqValDense[:,0] /= 2.
        freqValDense[:,nmesh[1]] = freqValDense[:,0]
    '''
    
    freqValDense = np.roll(freqValDense, (-(nmesh[0])//2, -(nmesh[1])//2), axis=(0,1))

    return freqValDense

def PadZeroForInterpolation3d(freqVal, nmesh, nDenseMesh):
    if nmesh[0] > nDenseMesh[0] or nmesh[1] > nDenseMesh[1] or nmesh[2] > nDenseMesh[2] :
        return freqVal
    freqVal = np.roll(freqVal, ((nmesh[0]+1)//2, (nmesh[1]+1)//2, (nmesh[2]+1)//2), axis=(0,1,2) )
    freqValDense = np.zeros(nDenseMesh, dtype=complex)
    freqValDense[0:nmesh[0],0:nmesh[1],0:nmesh[2]] = freqVal

    '''
    if nmesh[0]%2 == 0 and nDenseMesh[0] > nmesh[0]:
        freqValDense[0,:,:] /= 2.
        freqValDense[nmesh[0],:,:] = freqValDense[0,:,:]
    if nmesh[1]%2 == 0 and nDenseMesh[1] > nmesh[1]:
        freqValDense[:,0,:] /= 2.
        freqValDense[:,nmesh[1],:] = freqValDense[:,0,:]
    if nmesh[2]%2 == 0 and nDenseMesh[2] > nmesh[2]:
        freqValDense[:,:,0] /= 2.
        freqValDense[:,:,nmesh[2]] = freqValDense[:,:,0]
    '''
    
    freqValDense = np.roll(freqValDense, (-nmesh[0]//2, -nmesh[1]//2, -nmesh[2]//2), axis=(0,1,2))

    return freqValDense


