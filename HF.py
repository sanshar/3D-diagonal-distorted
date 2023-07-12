import pyscf, LindseyWavelets, Distort, time, scipy, Distort2
import numpy as np
from diis import FDiisContext
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, jacfwd, jacrev

import ctypes, numpy
ndpointer = numpy.ctypeslib.ndpointer
libLindsey = ctypes.cdll.LoadLibrary('libLindsey.so')
LindseyVals = libLindsey.LindseyVals
LindseyVals.restype = None
LindseyVals.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
]


def getKinetic(orbs, G2, nmesh, factor):
    ngrid, nbas = orbs.shape[1], orbs.shape[0]
    
    orbsG = (1.+0j)*orbs
    for i in range(nbas):
        orbsG[i] = np.fft.fftn(orbsG[i].reshape(nmesh)).flatten()/ngrid**0.5
    KE = orbsG.dot(np.einsum('gi,g->gi', orbsG.T.conj(), G2/2.)) 
    return KE

def getNuclearPot(grid, nucPos, Z):
    pot = 0.*grid[:,0]
    rloc, C1, C2, C3, C4 = 0.2, -4.18023680,  0.72507482, 0., 0.
    #rloc, C1, C2, C3, C4 = 0.20000000, -9.11202340, 1.69836797, 0., 0.
    for zi, pos in enumerate(nucPos):
        r = np.einsum('gi,gi->g', grid-pos, grid-pos)**0.5 + 1.e-8
        rfac = r/rloc
        pot += -Z[zi] * scipy.special.erf(rfac/2.**0.5)/r + np.exp(-rfac**2/2) * (C1 + C2 * rfac**2 + C3 * rfac**4 + C4 * rfac**6)
    return pot

def getV2e(orbs, G2, nmesh, J, factor):
    nbas, ngrid = orbs.shape[0], orbs.shape[1]

    G2[0] = 1.
    potG = np.pi*4./G2
    potG[0] = 0.
    
    orbsG = (1.+0j)*orbs
    for i in range(nbas):
        orbsG[i] = np.fft.fftn((J**0.5*orbsG[i]).reshape(nmesh)).flatten() * factor 
    V2e = orbsG.dot(np.einsum('g,ig->ig', potG, orbsG).conj().T) #* factor #* factor * ngrid
    return V2e

        
def getOrbVal(x, L):
    return LindseyWavelets.getVal(x) + LindseyWavelets.getVal(x+L) + LindseyWavelets.getVal(x-L)

##x is between 0 and 2pi
def basVal(x, N):
    xi = x+1.e-8
    mat  = 1./N * np.sin(N*xi/2) / np.tan(xi/2) + 1.j*np.sin(N*xi/2)/N    
    mat[abs(x)<1e-8] = 1.
    return mat

def basprimeVal(x, N):
    xi = x+1.e-8
    mat =  1/2 * np.cos(N*xi/2) / np.tan(xi/2)  - np.sin(N*xi/2)/np.sin(xi/2)**2/2/N + 1.j*np.cos(N*xi/2)/2.
    mat[abs(x)<1e-8] = 1.j/2.
    
    return mat

#'''
def getOrbsSlow(basgrid, grid, Jacobian, dxbas, dybas, dzbas, Lx, Ly, Lz):
    nbas, ngrid = basgrid.shape[0], grid.shape[0]

    orbs = np.zeros((nbas, ngrid))
    for i, baspt in enumerate(basgrid):
        print(i, nbas)
        orbs[i] = getOrbVal( (grid[:,0]-baspt[0])/dxbas, Lx/dxbas) / dxbas**0.5 \
            * getOrbVal( (grid[:,1]-baspt[1])/dybas, Ly/dybas) / dybas**0.5 \
            * getOrbVal( (grid[:,2]-baspt[2])/dzbas, Lz/dzbas) / dzbas**0.5 

        orbs[i] = orbs[i] * Jacobian**0.5
    return orbs
#'''

def getOrbs(basgrid, grid, basMesh, ax, bx, ay, by, az, bz):
    nbas, ngrid = basgrid.shape[0], grid.shape[0]

    orbs = np.zeros((nbas, ngrid))
    orbx, orby, orbz = np.zeros((ngrid,)), np.zeros((ngrid,)), np.zeros((ngrid,))
    gridx, gridy, gridz = grid[:,0], grid[:,1], grid[:,2]
    Lx, Ly, Lz = (bx-ax), (by-ay), (bz-az)
    
    orbval = np.zeros((nbas, ngrid), dtype=complex)
    for i in range(nbas):
        orbx = basVal( (gridx - basgrid[i,0]) * 2. * np.pi / Lx, basMesh[0])
        orby = basVal( (gridy - basgrid[i,1]) * 2. * np.pi / Ly, basMesh[1])
        orbz = basVal( (gridz - basgrid[i,2]) * 2. * np.pi / Lz, basMesh[2])

        orbval[i] = orbx*orby*orbz
    return orbval

def getOrbsDeriv(basgrid, grid, basMesh, ax, bx, ay, by, az, bz):
    nbas, ngrid = basgrid.shape[0], grid.shape[0]

    orbs = np.zeros((nbas, ngrid))
    orbx, orby, orbz = np.zeros((ngrid,)), np.zeros((ngrid,)), np.zeros((ngrid,))
    gridx, gridy, gridz = grid[:,0]+1e-8, grid[:,1]+1e-8, grid[:,2]+1e-8
    Lx, Ly, Lz = (bx-ax), (by-ay), (bz-az)
    
    orbval = np.zeros((nbas, ngrid, 3), dtype=complex)
    for i in range(nbas):
        orbx   = basVal( (gridx - basgrid[i,0]) * 2. * np.pi / Lx, basMesh[0])
        orby   = basVal( (gridy - basgrid[i,1]) * 2. * np.pi / Ly, basMesh[1])
        orbz   = basVal( (gridz - basgrid[i,2]) * 2. * np.pi / Lz, basMesh[2])

        orbxdx = basprimeVal( (gridx - basgrid[i,0]) * 2. * np.pi / Lx, basMesh[0]) * 2. * np.pi / Lx
        orbydy = basprimeVal( (gridy - basgrid[i,1]) * 2. * np.pi / Ly, basMesh[1]) * 2. * np.pi / Ly
        orbzdz = basprimeVal( (gridz - basgrid[i,2]) * 2. * np.pi / Lz, basMesh[2]) * 2. * np.pi / Lz

        orbval[i,:,0] = orbxdx*orby  *orbz
        orbval[i,:,1] = orbx  *orbydy*orbz
        orbval[i,:,2] = orbx  *orby  *orbzdz
    return orbval

def get_Gv(nmesh,cell):
    rx = np.fft.fftfreq(nmesh[0], 1./nmesh[0])
    ry = np.fft.fftfreq(nmesh[1], 1./nmesh[1])
    rz = np.fft.fftfreq(nmesh[2], 1./nmesh[2])

    return np.asarray(np.dot(pyscf.lib.cartesian_prod((rx,ry,rz)), cell.reciprocal_vectors()),dtype=np.double)

def makeJK(rdm, V2e, factor):
  if (len(V2e.shape) == 2):
    J = np.diag(V2e.dot(rdm.diagonal()) * factor).real
    K = V2e * rdm * factor
  else:
    J = np.einsum('ijkl,lk->ij', V2e, rdm)
    K = np.einsum('ijkl,kj->il', V2e, rdm)
      
  return J, K

def get_grad(orbs, nelec, fock):
    return np.dot(orbs[:,nelec//2:].T.conj(), fock.dot(orbs[:,:nelec//2])).flatten() 

def HF(cell, basMesh, nmesh, mf, eps = 1.e-6):
    a1,b1,a2,b2,a3,b3 = -cell.a[0,0]//2,cell.a[0,0]//2,-cell.a[1,1]//2,cell.a[1,1]//2,-cell.a[2,2]//2,cell.a[2,2]//2

    #grid = pyscf.lib.cartesian_prod([np.arange(x)/x for x in nmesh])
    #grid = np.dot(grid, cell.lattice_vectors())
    grid = cell.get_uniform_grids(nmesh)
    #basgrid = np.dot(pyscf.lib.cartesian_prod([np.arange(x)/x for x in basMesh]), cell.lattice_vectors())
    basgrid = cell.get_uniform_grids(basMesh)
    

    dxbas, dybas, dzbas = (b1-a1)/basMesh[0], (b2-a2)/basMesh[1], (b3-a3)/basMesh[2]
    C = np.fromfile("cheb.bin", dtype=np.float64).reshape(40, 40, 40, -1)

    if (C.shape[3]%3 != 0):
        print("the number of layers should be multiple of 3")
        exit(0)

    maxIter = 200
    alpha = 0.8
    tol = 1e-6
    Tbasx, Tbasy, Tbasz, Jinv2 = Distort.inv_flow(grid[:,0], grid[:,1], grid[:,2], C, a1, b1, a2, b2, a3, b3, maxIter, alpha, tol)
    distortedGrid = np.hstack((Tbasx.reshape(-1,1), Tbasy.reshape(-1,1), Tbasz.reshape(-1,1)))
    
    ngrid, nbas = np.prod(nmesh), np.prod(basMesh)
    Lx, Ly, Lz = b1-a1, b2-a2, b3-a3
    w, wbas = cell.vol/ngrid, (cell.vol/nbas)
    G = get_Gv(nmesh, cell)
    G2 = np.einsum('gi,gi->g', G, G)

    t0 = time.time()
    #import pdb
    #pdb.set_trace()
    #TT = Distort2.flow_vmap(distortedGrid+1e-6, C, a1,b1,a2,b2,a3,b3)

    Tdel = jacfwd(lambda grid : Distort2.flow(grid,C, a1,b1,a2,b2,a3,b3))
    Jac = vmap(Tdel, in_axes=(0))(distortedGrid+1e-6)
    print(time.time()-t0)

    JacGrad = Jac[:,3]  #Del |J|
    Jac = Jac[:,:3]     #J
    Jacinv = np.asarray([np.linalg.inv(Ji) for Ji in Jac])      #J
    JacDet = np.asarray([np.linalg.det(Ji) for Ji in Jac])  #|J|

    nucPos = np.asarray([cell._env[cell._atm[i,1]:cell._atm[i,1]+3] for i in range(cell._atm.shape[0])])

    orbvalCardinal          = np.eye(nbas, dtype=complex) 
    orbval                  = np.exp(-1.j*np.einsum('ai,gi->ga', grid, G))
    orbDerivValG            = np.einsum('ag,gi->agi', orbval, 1.j*G)
    orbDerivValCardinal     = np.zeros((nbas, nbas, 3), dtype=complex)

    for i in range(nbas):
        orbDerivValCardinal[i,:,0] = np.fft.ifftn(orbDerivValG[i,:,0].reshape(nmesh)).flatten()
        orbDerivValCardinal[i,:,1] = np.fft.ifftn(orbDerivValG[i,:,1].reshape(nmesh)).flatten()
        orbDerivValCardinal[i,:,2] = np.fft.ifftn(orbDerivValG[i,:,2].reshape(nmesh)).flatten()
    
    orbDerivValCardinal =  np.einsum('bji,abj->abi', Jac, orbDerivValCardinal) 

    KE = np.einsum('axi,bxi,x->ab', orbDerivValCardinal.conj(), orbDerivValCardinal, 1./JacDet)/2.
    KE = np.einsum('a,ab,b->ab', JacDet**0.5, KE, JacDet**0.5)

    vpot = getNuclearPot(distortedGrid, nucPos, cell._atm[:,0]) 

    Vne = np.diag(vpot)
    ##solve eigenvalue problem
    print(np.linalg.eigh(KE+Vne)[0][:10])



    alpha = 1.6

    ## solve poisson's equation
    grid2 = np.einsum('gi,gi->g', distortedGrid, distortedGrid)
    fval = np.exp(-grid2*alpha)

    [dke, vke] = np.linalg.eigh(2.*KE)
    L = 3.
    g2 = 1.*G2
    g2.sort()
    Vtrunc = 0.*dke
    #Vtrunc[1:] = 4.*np.pi/dke[1:]

    Vtrunc[1:] = 8.*np.pi*(np.sin(L*dke[1:]**0.5/2))**2/dke[1:]
    Vtrunc[0] = 8.*np.pi*(L/2)**2
    V = np.dot(vke, np.einsum('a,ab->ab', Vtrunc, vke.conj().T))
    V2e = np.einsum('a,ab,b->ab', 1./JacDet**0.5, V, 1./JacDet**0.5)
    potx = np.dot(V2e, fval)
    
    print("pot energy ", np.sum(fval.conj()*potx) * cell.vol/nbas)

    V2e = np.einsum('a,ab,b->ab', JacDet**0.5, V, JacDet**0.5)

    ###Potential
    '''
    potG = 2.*np.sin(L*G2**0.5/2)**2/G2
    potG[0] = L**2/2
    #potG = 1./G2
    #potG[0] = 0.
    potG *= 4.*np.pi
    
    grid2 = np.einsum('gi,gi->g', grid, grid)
    fval = np.exp(-grid2*alpha)
    fpot = np.fft.ifftn( (potG * np.fft.fftn(fval.reshape(nmesh)).flatten()).reshape(nmesh)).flatten()
    print("potential ", np.einsum('g,g', fpot.conj(), fval) * (cell.vol/nbas), (np.pi/2./alpha)**1.5 * (2*np.pi/0.5/alpha))
    '''

    dc = FDiisContext(5)

    CoreH, nelec, nucPot = Vne+KE, cell.nelectron, 0. #cell.energy_nuc()

    Fock = 1.*CoreH
    ##SCF
    for it in range(20):
        e, orbs = np.linalg.eigh(Fock)
        rdm = 2.*np.dot(orbs[:,:nelec//2], orbs[:,:nelec//2].T.conj())
        
        j, k = makeJK(rdm, V2e, nbas/cell.vol) 
        Fock = j -0.5*k + CoreH

        OrbGrad = get_grad(orbs, nelec, Fock)
        fOrbGrad = np.linalg.norm(OrbGrad)

        Energy = np.einsum('ij,ji', (0.5*j-0.25*k+CoreH), rdm) + nucPot 
        print("{0:3d}   {1:15.8f}  {2:15.8f}".format(it, Energy.real, fOrbGrad), e[0] )
        if (fOrbGrad < 1.e-5):
           break

        Fock, OrbGrad, c0 = dc.Apply(Fock, OrbGrad)
        
    return Energy.real , orbs[:,:nelec//2].T   

