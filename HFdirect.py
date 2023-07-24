import pyscf, LindseyWavelets, Distort, time, scipy, Distort2
import numpy as np
from diis import FDiisContext
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, jacfwd, jacrev
import finufft, FFTInterpolation
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


def getNuclearPotDenseGrid(denseMesh, nbas, nucPos, Z, invflow, flowJac, cell):
    grid = cell.get_uniform_grids(denseMesh)
    distortedGrid = invflow(grid)
    JacAll = flowJac(distortedGrid+1.e-5)
    
    Jac = JacAll[:,:3]     #J
    JacDet = np.asarray([np.linalg.det(Ji) for Ji in Jac]) 
    
    nmesh = cell.mesh
    cell.mesh = denseMesh
    SI = cell.get_SI()
    Gv = cell.get_Gv(denseMesh)
    vpplocG = pyscf.pbc.gto.pseudo.get_vlocG(cell, Gv)
    vpplocG = -numpy.einsum('ij,ij->j', SI, vpplocG)

    gridfft = np.array(distortedGrid*2*np.pi/cell.a[0,0], dtype='float64')
    nucPot1 = finufft.nufft3d2(1.*gridfft[:,0], 1.*gridfft[:,1], 1.*gridfft[:,2], vpplocG.reshape(denseMesh),modeord=1,isign=1).real


    nucPot = np.zeros((nbas,))
    orbvalCardinal          = np.eye(nbas, dtype=complex) 
    for i in range(nbas):
        orbDense = np.fft.ifftn(FFTInterpolation.PadZeroForInterpolation3d(np.fft.fftn(orbvalCardinal[i].reshape(nmesh)), nmesh, denseMesh)).flatten()
        nucPot[i] = orbDense.conj().dot(nucPot1/JacDet**0.5).real/cell.vol

    cell.mesh = nmesh
    return nucPot
    
def getNuclearPot(grid, nucPos, Z):
    pot = 0.*grid[:,0]
    rloc, C1, C2, C3, C4 = 0.2, -4.18023680,  0.72507482, 0., 0.
    #rloc, C1, C2, C3, C4 = 0.20000000, -9.11202340, 1.69836797, 0., 0.
    for zi, pos in enumerate(nucPos):
        r = np.einsum('gi,gi->g', grid-pos, grid-pos)**0.5 + 1.e-8
        rfac = r/rloc
        pot += -Z[zi] * scipy.special.erf(rfac/2.**0.5)/r + np.exp(-rfac**2/2) * (C1 + C2 * rfac**2 + C3 * rfac**4 + C4 * rfac**6)
    return pot

def getNuclearPotShortRange(grid, nucPos, Z, JacDet):
    pot = 0.*grid[:,0]
    rloc, C1, C2, C3, C4 = 0.2, -4.18023680,  0.72507482, 0., 0.

    p = 1./rloc**2/2.
    
    nucDensity = 0.*pot
    #rloc, C1, C2, C3, C4 = 0.20000000, -9.11202340, 1.69836797, 0., 0.
    for zi, pos in enumerate(nucPos):
        r = np.einsum('gi,gi->g', grid-pos, grid-pos)**0.5 #+ 1.e-8
        rfac = r/rloc
        pot += np.exp(-rfac**2/2) * (C1 + C2 * rfac**2 + C3 * rfac**4 + C4 * rfac**6)
                
        nucDensity += -Z[zi] * (p/np.pi)**1.5 * np.exp(-p * r**2)
    return pot, nucDensity

def getNuclearPotLongRangeRealSpace(grid, nucPos, Z):
    pot = 0.*grid[:,0]
    rloc, C1, C2, C3, C4 = 0.2, -4.18023680,  0.72507482, 0., 0.
    #rloc, C1, C2, C3, C4 = 0.20000000, -9.11202340, 1.69836797, 0., 0.
    for zi, pos in enumerate(nucPos):
        r = np.einsum('gi,gi->g', grid-pos, grid-pos)**0.5 + 1.e-8
        rfac = r/rloc
        pot += -Z[zi] * scipy.special.erf(rfac/2.**0.5)/r 
    return pot

def getNuclearDensity(grid, nucPos, Z, cell, basMesh):
    pot = 0.*grid[:,0]
    rloc = 0.2
    p = 1./rloc**2/2.

    L = cell.a[0,0]
    ##get potential due to this gaussian on a uniform grid
    nmesh = [45,45,45]
    G = get_Gv(nmesh, cell)
    G2 = np.einsum('gi,gi->g', G, G)
    G2[0] = 1.
    
    for zi, pos in enumerate(nucPos):
        fG = -Z[zi]*(1+0.j)*np.exp(-G2/4./p) * np.exp(1.j*np.einsum('Gi,i->G', G, pos))
        #vG = fG * 8.*np.pi*(np.sin(L*G2**0.5/2))**2/G2
        #vG[0] = fG[0] * 8.*np.pi*(L/2)**2

        vG = fG * 4.*np.pi/G2
        vG[0] = 0.


    grid = grid * 2 * np.pi/cell.a[0,0]  ###this should be different
    #pot = finufft.nufft3d2(1.*grid[:,0], 1.*grid[:,1], 1.*grid[:,2], vG.reshape(nmesh))
    pot = finufft.nufft3d2(1.*grid[:,0], 1.*grid[:,1], 1.*grid[:,2], vG.reshape(nmesh),modeord=1)/cell.vol

    return pot

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

##it is the non-symmetric version
def applyKE(fval, Jac, G, nmesh):
    nbas = Jac.shape[0]
    fvalD = np.einsum('gi,g->ig',1.j*G, np.fft.fftn(fval.reshape(nmesh)).flatten())
    for i in range(3):
        fvalD[i] = np.fft.ifftn(fvalD[i].reshape(nmesh)).flatten()
    fvalD = np.einsum('bji, jb->ib', Jac, fvalD)
    
    fval2D = np.zeros((3,3,nbas), dtype=complex)
    for a in range(3):
        fval2D[a] = np.einsum('gi,g->ig',1.j*G, np.fft.fftn(fvalD[a].reshape(nmesh)).flatten())
        for j in range(3):
            fval2D[a,j] = np.fft.ifftn(fval2D[a,j].reshape(nmesh)).flatten()
    
    fval2D = np.einsum('bji, ijb->b', Jac, fval2D)
    return fval2D


##USE FFT to get Kinetic inv on vector O(N)
def Condition(v, G2, basMesh):

    v0 = np.fft.fftn(v.reshape(basMesh)).flatten()
    v0 = v0/G2
    v0[0] = 0.
    v0 = np.fft.ifftn(v0.reshape(basMesh)).flatten()

    return v0

##Use FFT to get gradient of vector \nabla Psi
def HvGrad(v, JacDet, Jac, basMesh, G):
    v0 = 1./JacDet**0.5
    v0 = v0/np.linalg.norm(v0)
    
    v = v - v0.dot(v) * v0

    Ag = np.einsum('g,gi->ig', np.fft.fftn(v.reshape(basMesh)).flatten(), 1.j*G)
    for i in range(3):
        Ag[i] = np.fft.ifftn(Ag[i].reshape(basMesh)).flatten()

    A = np.einsum('bji,jb->bi', Jac, Ag)

    return A

##USE FFT to act Kinetic energy on vector O(N)
def Hv2(v, JacDet, Jac, basMesh, G):
    v0 = 1./JacDet**0.5
    v0 = v0/np.linalg.norm(v0)
    
    v = v - v0.dot(v) * v0
    
    A  = JacDet**0.5 * v
    Ag = np.einsum('g,gi->ig', np.fft.fftn(A.reshape(basMesh)).flatten(), 1.j*G)
    for i in range(3):
        Ag[i] = np.fft.ifftn(Ag[i].reshape(basMesh)).flatten()

    A = np.einsum('bji,jb->bi', Jac, Ag)
    A = np.einsum('bi,b->bi', A, 1./JacDet)   
    A = np.einsum('bij,bj->ib', Jac, A)    

    out = (0.+0.j)*v
    for i in range(3):
        out -= np.fft.ifftn( (np.fft.fftn(A[i].reshape(basMesh)).flatten()*1.j*G[:,i]).reshape(basMesh) ).flatten()

    Hv = out * JacDet**0.5

    Hv = Hv - v0.dot(Hv) * v0

    return Hv

##USE Mat x vec Kinetic energy on vector O(N^2)
def Hv(v, JacDet, orbDerivValCardinal):
    v0 = 1./JacDet**0.5
    v0 = v0/np.linalg.norm(v0)
    
    v = v - v0.dot(v) * v0
    
    A  = JacDet**0.5 * v


    A  = np.einsum('bxi, b->xi', orbDerivValCardinal, A)
    A  = np.einsum(' xi, x->xi', A, 1./JacDet)
    Hv = np.einsum('axi, xi->a', orbDerivValCardinal.conj(), A) * JacDet**0.5

    Hv = Hv - v0.dot(Hv) * v0
    #Hv = Hv + 1.e3 * v0
    
    return Hv

def HF(cell, basMesh, nmesh, mf, eps = 1.e-6):
    a1,b1,a2,b2,a3,b3 = 0.,cell.a[0,0],0.,cell.a[1,1],0.,cell.a[2,2]

    grid = cell.get_uniform_grids(nmesh)
    basgrid = cell.get_uniform_grids(basMesh)
    

    C = np.fromfile("cheb.bin", dtype=np.float64).reshape(10, 10, 10, -1)

    if (C.shape[3]%3 != 0):
        print("the number of layers should be multiple of 3")
        exit(0)

    maxIter = 200
    alpha = 0.8
    tol = 1e-8
    
    def invFlow(grid):
        Tbasx, Tbasy, Tbasz, Jinv2 = Distort.inv_flow(grid[:,0], grid[:,1], grid[:,2], C, a1, b1, a2, b2, a3, b3, maxIter, alpha, tol)
        return np.hstack((Tbasx.reshape(-1,1), Tbasy.reshape(-1,1), Tbasz.reshape(-1,1)))
    distortedGrid = invFlow(grid)
    #distortedGrid = Distort2.invflow3(grid, C, a1, b1, a2, b2, a3, b3)
    
    
    ngrid, nbas = np.prod(nmesh), np.prod(basMesh)
    G = get_Gv(nmesh, cell)
    G2 = np.einsum('gi,gi->g', G, G)

    t0 = time.time()


    Tdel = jacfwd(lambda grid : Distort2.flow(grid,C, a1,b1,a2,b2,a3,b3))
    JacAll = vmap(Tdel, in_axes=(0))(distortedGrid+1.e-5)
    #Tdel = jacfwd(lambda grid : Distort2.flow3(grid,C, a1,b1,a2,b2,a3,b3))
    #JacAll = vmap(Tdel, in_axes=(0))(distortedGrid)


    Jac = JacAll[:,:3]     #J
    JacDet = np.asarray([np.linalg.det(Ji) for Ji in Jac])  #|J|
    JacGrad = JacAll[:,3]  #Del |J|

    
    from scipy.sparse.linalg import LinearOperator    
    G2[0] = 1.
    Hvop2   = LinearOperator((nbas, nbas), matvec = lambda v: Hv2(v, JacDet, Jac, basMesh, G))        
    preCond = LinearOperator((nbas, nbas), matvec = lambda v: Condition(v, G2, basMesh))
    printNorm = lambda v : print(np.linalg.norm(v))

    v0 = 1./JacDet**0.5
    v0 = v0/np.linalg.norm(v0)


    dc = FDiisContext(5)

    nucPos = np.asarray([cell._env[cell._atm[i,1]:cell._atm[i,1]+3] for i in range(cell._atm.shape[0])])


    denseMesh = [40,40,40]
    nucPot = getNuclearPotDenseGrid(denseMesh, nbas, nucPos, cell._atm[:,0], \
        invFlow, vmap(Tdel, in_axes=(0)), cell) * JacDet**0.5


    alpha = 1.805868146
    grid2 = np.einsum('gi,gi->g', distortedGrid-np.asarray([3.,3.,3.]), distortedGrid-np.asarray([3.,3.,3.]))
    fval = np.exp(-(grid2)*alpha)/JacDet**0.5

    o = np.sum(fval.conj()*fval)*cell.vol/nbas
        
    kef = Hv2(fval, JacDet, Jac, basMesh, G)/2.
    print(np.sum(kef.conj()*fval)*cell.vol/nbas/o, o)


    vef = nucPot*fval
    print(np.sum(vef.conj()*fval)*cell.vol/nbas/o, o)
    
    import pdb
    pdb.set_trace()

    from scipy.sparse.linalg import eigsh    
    Ham   = LinearOperator((nbas, nbas), matvec = lambda v: nucPot*v + Hv2(v, JacDet, Jac, basMesh, G)/2.)        
    (w,v) = eigsh(Ham, 1, which = 'SA')
    
    print(w[0], cell.energy_nuc())
    
    exit(0)
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


    return
    
    #potx  = scipy.sparse.linalg.cg(Hvop , fval, M=preCond, callback=printNorm)[0] * 4. * np.pi

    '''
    KE3 = np.zeros((nbas, nbas), dtype=complex)
    KE2 = np.zeros((nbas, nbas), dtype=complex)
    for i in range(nbas):
        temp = 0.*fval
        temp[i] = 1.
        tempKE = applyKE(temp, Jac, G, nmesh)
        KE3[:,i] = -tempKE/JacDet
    '''
    
    
    fvalD = np.einsum('gi,g->ig',1.j*G, np.fft.fftn(fval.reshape(nmesh)).flatten())
    for i in range(3):
        fvalD[i] = np.fft.ifftn(fvalD[i].reshape(nmesh)).flatten()
    fvalD = np.einsum('bji, jb->ib', Jac, fvalD)
    
    fval2D = np.zeros((3,3,nbas), dtype=complex)
    for a in range(3):
        fval2D[a] = np.einsum('gi,g->ig',1.j*G, np.fft.fftn(fvalD[a].reshape(nmesh)).flatten())
        for j in range(3):
            fval2D[a,j] = np.fft.ifftn(fval2D[a,j].reshape(nmesh)).flatten()
    
    fval2D = np.einsum('bji, ijb->b', Jac, fval2D)

    print("kinetic1 ", np.sum(fval2D * fval/JacDet).real * (cell.vol/nbas), (np.pi*alpha/2)**0.5*(np.pi/2/alpha)*3)
    print("kinetic2 ", np.einsum('ib,ib,b', fvalD.conj(), fvalD, 1./JacDet).real * (cell.vol/nbas), (np.pi*alpha/2)**0.5*(np.pi/2/alpha)*3)

    fval2 = fval/JacDet**0.5
    fvalD2 = np.einsum('gi,g->ig',1.j*G, np.fft.fftn(fval2.reshape(nmesh)).flatten())
    for i in range(3):
        fvalD2[i] = np.fft.ifftn(fvalD2[i].reshape(nmesh)).flatten()

    fvalD2 = np.einsum('bji, jb->ib', Jac, fvalD2)
    phiJ = np.einsum('b,bi->bi', fval2, JacGrad)
    JacGrd2 = np.einsum('bi,bi->b', JacGrad, JacGrad)
    ke3 = np.einsum('ib,ib', fvalD2.conj(), fvalD2) +  np.sum(JacGrd2*fval2*fval2/JacDet**2)/4. +\
        (np.einsum('bi,ib,b', phiJ.conj(), fvalD2, 1./JacDet)).real
    print("kinetic3 ", ke3.real * (cell.vol/nbas), (np.pi*alpha/2)**0.5*(np.pi/2/alpha)*3)

    #print(np.max(abs(KE3 - KE3.conj().T)))
    #wt = cell.vol/nbas
    #print(np.einsum('ib,ib', fvalD2.conj(), fvalD2)*wt,np.sum(JacGrd2*fval2*fval2/JacDet**2)/4.*wt, (np.einsum('bi,ib,b', phiJ.conj(), fvalD2, 1./JacDet)).real*wt )

    #return 

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

    KE1 = np.einsum('axi,bxi,x->ab', orbDerivValCardinal.conj(), orbDerivValCardinal, 1./JacDet)
    
    KE = np.einsum('axi,bxi->ab', orbDerivValCardinal.conj(), orbDerivValCardinal)
    t1 = fval2.dot(KE.dot(fval2))*cell.vol/nbas

    term2 = (0.5*np.einsum('abi,bi,b->ab', orbDerivValCardinal.conj(), JacGrad, 1./JacDet ))
    t2 = fval2.dot((term2+term2.T.conj()).dot(fval2))*cell.vol/nbas
    KE += term2+term2.T.conj()

    term3 = 0.25 * np.diag(np.einsum('bi,bi,b->b', JacGrad.conj(), JacGrad, 1./JacDet**2))
    t3 = fval2.dot((term3).dot(fval2))*cell.vol/nbas
    KE += term3
    
    print(fval2.dot(KE.dot(fval2)) * cell.vol/nbas)
    print(t1, t2, t3)
    print()

    #KE *= 0.5

    #KE2 = np.einsum('axi,bxi,x->ab', orbDerivValCardinal.conj(), orbDerivValCardinal, 1./JacDet)/2.
    KE = np.einsum('a,ab,b->ab', JacDet**0.5, KE1, JacDet**0.5)
 
    vpot = getNuclearPot(distortedGrid, nucPos, cell._atm[:,0]) 

    Vne = np.diag(vpot)

    KE33 = np.einsum('a,ab,b->ab', JacDet**0.5, KE, JacDet**0.5)
    ##solve eigenvalue problem
    print(np.linalg.eigh(KE33/2.+Vne)[0][:10])
    print("\n")


    ## solve poisson's equation
    grid2 = np.einsum('gi,gi->g', distortedGrid-np.asarray([3.,3.,3.]), distortedGrid-np.asarray([3.,3.,3.]))
    #grid2 = np.einsum('gi,gi->g', distortedGrid, distortedGrid)
    fval = np.exp(-grid2*alpha)/JacDet**0.5

    [dke, vke] = np.linalg.eigh(KE)
    
    L = cell.a[0,0]/2.
    g2 = 1.*G2
    g2.sort()
    Vtrunc = 0.*dke
    Vtrunc[1:] = 4.*np.pi/dke[1:]

    v0 = 1./JacDet**0.5
    v0 = v0/np.linalg.norm(v0)
    
    from scipy.sparse.linalg import LinearOperator    
    G2[0] = 1.
    Hvop    = LinearOperator((nbas, nbas), matvec = lambda v: Hv(v, JacDet, orbDerivValCardinal))        
    Hvop2   = LinearOperator((nbas, nbas), matvec = lambda v: Hv2(v, JacDet, Jac, basMesh, G))        
    preCond = LinearOperator((nbas, nbas), matvec = lambda v: Condition(v, G2, basMesh))
    printNorm = lambda v : print(np.linalg.norm(v))

    fval = fval - fval.dot(v0) * v0

    import pdb
    pdb.set_trace()
    
    potx2 = scipy.sparse.linalg.cg(Hvop2, fval, M=preCond, callback=printNorm)[0] * 4. * np.pi
    #potx2 = scipy.sparse.linalg.cg(Hvop, fval, M=preCond, callback=printNorm)[0] * 4. * np.pi


    #Vtrunc[1:] = 8.*np.pi*(np.sin(L*dke[1:]**0.5/2))**2/dke[1:]
    #Vtrunc[0] = 8.*np.pi*(L/2)**2
    V = np.dot(vke, np.einsum('a,ab->ab', Vtrunc, vke.conj().T))
    potx = np.dot(V, fval)
    
    print("pot energy ", np.sum(fval.conj()*potx) * cell.vol/nbas, (np.pi/2./alpha)**1.5 * (2*np.pi/0.5/alpha))
    print("pot energy ", np.sum(fval.conj()*potx2) * cell.vol/nbas, (np.pi/2./alpha)**1.5 * (2*np.pi/0.5/alpha))

    import pdb
    pdb.set_trace()
    return

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

