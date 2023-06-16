import pyscf, LindseyWavelets, Distort, time, scipy
import numpy as np
from diis import FDiisContext
import matplotlib.pyplot as plt

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
        orbsG[i] = np.fft.fftn(orbsG[i].reshape(nmesh)).flatten()*factor
    KE = orbsG.dot(np.einsum('gi,g->gi', orbsG.T.conj(), G2/2.)) 
    return KE

def getNuclear(orbs, SF, G2, nmesh, w):
    G2[0] = 1.0
    
    potG = np.pi*4.*SF/G2
    potG[0] = 0.
    potR = np.fft.ifftn(potG.reshape(nmesh)).flatten() 
    return orbs.dot(np.einsum('i,ij->ij',potR,orbs.T)) 

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

def getOrbs(basgrid, grid, Jacobian, dxbas, dybas, dzbas, Lx, Ly, Lz):
    nbas, ngrid = basgrid.shape[0], grid.shape[0]

    orbs = np.zeros((nbas, ngrid))
    orbx, orby, orbz = np.zeros((ngrid,)), np.zeros((ngrid,)), np.zeros((ngrid,))

    for i, baspt in enumerate(basgrid):
        orbx *= 0.
        orby *= 0.
        orbz *= 0.
        
        #import pdb
        #pdb.set_trace()
        #print(i, nbas)
        LindseyVals( (grid[:,0]-baspt[0])/dxbas, ngrid, orbx) 
        LindseyVals( (grid[:,0]-baspt[0] + Lx)/dxbas, ngrid, orbx) 
        LindseyVals( (grid[:,0]-baspt[0] - Lx)/dxbas, ngrid, orbx) 
        orbx = orbx/ dxbas**0.5

        LindseyVals( (grid[:,1]-baspt[1])/dybas, ngrid, orby) 
        LindseyVals( (grid[:,1]-baspt[1] + Ly)/dybas, ngrid, orby) 
        LindseyVals( (grid[:,1]-baspt[1] - Ly)/dybas, ngrid, orby) 
        orby = orby/ dybas**0.5

        LindseyVals( (grid[:,2]-baspt[2])/dzbas, ngrid, orbz) 
        LindseyVals( (grid[:,2]-baspt[2] + Lz)/dzbas, ngrid, orbz) 
        LindseyVals( (grid[:,2]-baspt[2] - Lz)/dzbas, ngrid, orbz) 
        orbz = orbz/ dzbas**0.5

        orbs[i] = orbx * orby * orbz * Jacobian**0.5
        '''        
        orbs[i] = getOrbVal( (grid[:,0]-baspt[0])/dxbas, Lx/dxbas) / dxbas**0.5 \
            * getOrbVal( (grid[:,1]-baspt[1])/dybas, Ly/dybas) / dybas**0.5 \
            * getOrbVal( (grid[:,2]-baspt[2])/dzbas, Lz/dzbas) / dzbas**0.5 

        orbs[i] = orbs[i] * Jacobian**0.5
        '''
    return orbs

def get_Gv(nmesh,cell):
    rx = np.fft.fftfreq(nmesh[0], 1./nmesh[0])
    ry = np.fft.fftfreq(nmesh[1], 1./nmesh[1])
    rz = np.fft.fftfreq(nmesh[2], 1./nmesh[2])

    return np.asarray(np.dot(pyscf.lib.cartesian_prod((rx,ry,rz)), cell.reciprocal_vectors()),dtype=np.double)

def makeJK(rdm, V2e, factor):
  if (len(V2e.shape) == 2):
    J = np.diag(V2e.dot(rdm.diagonal()) * factor)
    K = V2e * rdm * factor
  else:
    J = np.einsum('ijkl,lk->ij', V2e, rdm)
    K = np.einsum('ijkl,kj->il', V2e, rdm)
      
  return J, K

def get_grad(orbs, nelec, fock):
    return np.dot(orbs[:,nelec//2:].T, fock.dot(orbs[:,:nelec//2])).flatten() 

def HF(cell, basMesh, nmesh, mf):
    a1,b1,a2,b2,a3,b3 = 0,cell.a[0,0],0,cell.a[1,1],0,cell.a[2,2]

    grid = pyscf.lib.cartesian_prod([np.arange(x)/x for x in nmesh])
    grid = np.dot(grid, cell.lattice_vectors())
    #grid = cell.get_uniform_grids(nmesh)
    basgrid = np.dot(pyscf.lib.cartesian_prod([np.arange(x)/x for x in basMesh]), cell.lattice_vectors())
    #basgrid = cell.get_uniform_grids(basMesh)

    dxbas, dybas, dzbas = (b1-a1)/basMesh[0], (b2-a2)/basMesh[1], (b3-a3)/basMesh[2]
    C = np.fromfile("cheb.bin", dtype=np.float64).reshape(20, 20, 20, -1)
    
    ngrid, nbas = np.prod(nmesh), np.prod(basMesh)
    Lx, Ly, Lz = b1-a1, b2-a2, b3-a3
    w, dxbas = cell.vol/ngrid, (cell.vol/nbas)
    G = get_Gv(nmesh, cell)
    G2 = np.einsum('gi,gi->g', G, G)

    #print(C.shape)
    #T1, T2, T3, J = Distort.flow(grid[:,0], grid[:,1], grid[:,2], C, a1,b1,a2,b2,a3,b3)
    #distortedGrid = np.hstack((T1.reshape(-1,1), T2.reshape(-1,1), T3.reshape(-1,1)))
    J = np.ones((ngrid,))/cell.vol*w
    distortedGrid = 1.*grid

    #for i in range(1):
    #    fig, ax = plt.subplots()
    #    ax.scatter(T2[i*100:(i+1)*100],T3[i*100:(i+1)*100])
    #    plt.show()

    #orbs = getOrbs(basgrid[:10], distortedGrid, J/w*cell.vol, dxbas, dybas, dzbas, Lx, Ly, Lz)
    #orbs2 = getOrbsSlow(basgrid[:10], distortedGrid, J/w*cell.vol, dxbas, dybas, dzbas, Lx, Ly, Lz)

    #'''
    #S = np.dot(orbs, orbs.T)*w
    #print(np.max(abs(S - np.eye(10))))

    '''
    import pdb
    pdb.set_trace()
    orbs1 = getOrbs(basgrid[:10], grid, np.ones((ngrid,)), dxbas, dybas, dzbas, Lx, Ly, Lz)
    S1 = np.dot(orbs1, orbs1.T)*w
    print(np.max(abs(S1 - np.eye(10))))
    orbs2 = getOrbsSlow(basgrid[:10], grid, np.ones((ngrid,)), dxbas, dybas, dzbas, Lx, Ly, Lz)
    S2 = np.dot(orbs2, orbs2.T)*w
    print(np.max(abs(S2 - np.eye(10))))
    pdb.set_trace()
    '''
    #maxIter = 1000 # maximum number of iterations
    #alpha = 0.8 # mixing parameter
    #tol = 1e-12 # convergence tolerance
    #S1,S2,S3,_ = Distort.inv_flow(grid[:,0],grid[:,1],grid[:,2],C,a1,b1,a2,b2,a3,b3,maxIter,alpha,tol)
    #import pdb
    #pdb.set_trace()
    #'''

    nx = nmesh[0]
    orbx = np.zeros((nx,))
    gridx = 1.*grid[:nx,2]
    import pdb
    pdb.set_trace()
    baspt = 0.
    LindseyVals( (gridx-baspt)/dxbas, nx, orbx)
    #orbx = getOrbVal( (gridx-baspt)/dxbas, Lx/dxbas)
    LindseyVals( (gridx-baspt+Lx)/dxbas, nx, orbx)
    LindseyVals( (gridx-baspt-Lx)/dxbas, nx, orbx)
    
    plt.plot(gridx, orbx)
    plt.show()
    print(orbx.dot(orbx.T) * (w**(1./3.))/dxbas)
    import pdb
    pdb.set_trace()
    
    #orbs = getOrbs(basgrid, distortedGrid, J/w*cell.vol, dxbas, dybas, dzbas, Lx, Ly, Lz)
    orbs = getOrbs(basgrid, grid, np.ones((ngrid,)), dxbas, dybas, dzbas, Lx, Ly, Lz)
    #orbs = getOrbs(basgrid, distortedGrid, J/w*cell.vol, dxbas, dybas, dzbas, Lx, Ly, Lz)
    S = np.dot(orbs, orbs.T)*w
    print("ovlp error ", np.max(abs(S - np.eye(basgrid.shape[0]))))
    #orbs2 = getOrbsSlow(basgrid, distortedGrid, J/w*cell.vol, dxbas, dybas, dzbas, Lx, Ly, Lz)

    import pdb
    pdb.set_trace()
    #orbs3 = np.fromfile("orbs.tmp").reshape(nbas, ngrid)
    #J = np.fromfile("Jacobian.tmp").reshape(ngrid)
    #S = np.dot(orbs, orbs.T)*w
    #print(np.max(abs(S - np.eye(basgrid.shape[0]))))
    
    KE = getKinetic(orbs, G2, nmesh, w).real/cell.vol
    SF = cell.get_SI()
    
    '''
    ao = cell.pbc_eval_gto('GTOval', grid).T
    S2 = ao.T.dot(ao) * w
    S = cell.pbc_intor('cint1e_ovlp_sph')
    
    K = getKinetic(ao, G2, nmesh, w).real/cell.vol
    N = getNuclear(ao, np.dot(-cell.atom_charges(), SF), G2, nmesh, w)
    import pdb
    pdb.set_trace()
    [d,mo] = scipy.linalg.eigh(K+N, S)
    mydf = pyscf.pbc.df.FFTDF(cell)
    eri = mydf.get_eri()    
    rdm = 2.*np.dot(mo[:,:nelec//2], mo[:,:nelec//2].T.conj())
    print(np.sum(rdm, N), np.sum(rdm, K))
    '''

    Nuc = getNuclear(orbs, np.dot(-cell.atom_charges(), SF), G2, nmesh, w).real
    V2e = getV2e(orbs, G2, nmesh, J*cell.vol/w, w).real/dxbas/cell.vol
    #Nuc = getNuclear(orbs, np.ones((ngrid,)), G2, nmesh, w)
    [d,v] = np.linalg.eigh(KE+Nuc)
    print(d[:5])


    '''    
    ao = cell.pbc_eval_gto('GTOval', grid)
    mydf = pyscf.pbc.df.FFTDF(cell)
    eri = mydf.get_eri()    
    
    ao2G = (1.+0.j)*ao[:,1]*ao[:,1]
    ao2G = np.fft.fftn(ao2G.reshape(nmesh)).flatten()*w
    G2[0] = 1.
    potG = np.pi*4./G2
    potG[0] = 0.
    eri2 = np.sum(ao2G * potG * ao2G.conj())/cell.vol
    import pdb
    pdb.set_trace()
    '''
    dc = FDiisContext(5)

    CoreH, nelec, nucPot = Nuc+KE, cell.nelectron, cell.energy_nuc()
    print(nucPot, "nuclear pot")
    Fock = 1.*CoreH
    ##SCF
    for it in range(20):
        e, orbs = np.linalg.eigh(Fock)
        rdm = 2.*np.dot(orbs[:,:nelec//2], orbs[:,:nelec//2].T.conj())
        
        j, k = makeJK(rdm, V2e, 1.) 
        Fock = j -0.5*k + CoreH

        OrbGrad = get_grad(orbs, nelec, Fock)
        fOrbGrad = np.linalg.norm(OrbGrad)

        Energy = np.einsum('ij,ji', (0.5*j-0.25*k+CoreH), rdm) + nucPot 
        print("{0:3d}   {1:15.8f}  {2:15.8f}".format(it, Energy.real, fOrbGrad), e[0] )
        if (fOrbGrad < 1.e-5):
           #print(2.*np.einsum('ia,ij,ja->a', orbs[:,:nelec//2], Nuc, orbs[:,:nelec//2]))
           #print(2.*np.einsum('ia,ij,ja->a', orbs[:,:nelec//2], KE, orbs[:,:nelec//2]))
           #print(np.einsum('ia,ij,ja->a', orbs[:,:nelec//2], j, orbs[:,:nelec//2]))
           #print(np.einsum('ia,ij,ja->a', orbs[:,:nelec//2], k, orbs[:,:nelec//2]))
           break

        Fock, OrbGrad, c0 = dc.Apply(Fock, OrbGrad)
        
    return Energy.real , orbs[:,:nelec//2].T   

    
    