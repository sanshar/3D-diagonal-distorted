import pyscf, time, scipy, Distort2, Distort3
import numpy as np
from diis import FDiisContext
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, jacfwd, jacrev
import FFTInterpolation, Davidson, functools
import ctypes, numpy, time
from jax import numpy as jnp

'''
ndpointer = numpy.ctypeslib.ndpointer
libLindsey = ctypes.cdll.LoadLibrary('libLindsey.so')
LindseyVals = libLindsey.LindseyVals
LindseyVals.restype = None
LindseyVals.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
]
'''

def cond_and_body(y, invFlow, maxIter, alpha,tol):
    #@jit
    def cond_fun(carry):
        X0, X1, X2, T0, T1, T2, iter = carry
        #print(iter, X0, X1, X2)
      
        T0, T1, T2 = invFlow(X0, X1, X2) 

        relErr1 = np.abs(T0-y[0])
        relErr2 = np.abs(T1-y[1])
        relErr3 = np.abs(T2-y[2])
        cond1 = relErr1 > tol
        cond2 = relErr2 > tol
        cond3 = relErr3 > tol
        cond4 = iter < maxIter
        return (cond1+cond2+cond3)*cond4

        return jnp.max( T0-y[0], jnp.max(T1-y[1], T2-y[2])) > tol

    #@jit
    def body_fun(carry):   
        X0, X1, X2, T0, T1, T2, iter = carry
        X0 = X0 - alpha*(T0 - y[0])
        X1 = X1 - alpha*(T1 - y[1])
        X2 = X2 - alpha*(T2 - y[2])
        T0, T1, T2 = invFlow(X0, X1, X2)
        return (X0, X1, X2, T0, T1, T2, iter+1)
    return cond_fun, body_fun

def forwardFlow(invFlow, y):

    X = 0*y

    for i in range(y.shape[0]):
        cond_fun, body_fun = cond_and_body(y[i], invFlow, 100, 0.3, 1.e-8)
        T0, T1, T2 = invFlow(y[i,0], y[i,1], y[i,2])
        init_val = (y[i,0], y[i,1], y[i,2], T0, T1, T2, 0)
        
        
        val = init_val
        while cond_fun(val):
            val = body_fun(val)   
        X[i,0], X[i,1], X[i,2],_,_,_,iter = val
        #X[i,0], X[i,1], X[i,2],_,_,_,iter = lax.while_loop(cond_fun, body_fun, init_val)

    return X


def getStructureFactor(nbas, flow, Jacfun, invflow, cell):

    nucPos = np.asarray([cell._env[cell._atm[i,1]:cell._atm[i,1]+3] for i in range(cell._atm.shape[0])])

    Tx, Ty, Tz, Jac = flow(nucPos[:,0], nucPos[:,1], nucPos[:,2])
    Tc = np.vstack((Tx, Ty, Tz)).T
    Tc = Tc.reshape(-1,3)

    
    ##check error
    t1, t2, t3, _ = invflow(Tc[:,0], Tc[:,1], Tc[:,2])
    err1 = jnp.max(jnp.abs(t1-nucPos[:,0]))#/(b1-a1)
    err2 = jnp.max(jnp.abs(t2-nucPos[:,1]))#/(b2-a2)
    err3 = jnp.max(jnp.abs(t3-nucPos[:,2]))#/(b3-a3)
    
    print("Error in nuclear fit ", err1, err2, err3)


    #Tc = forwardFlow(invFlowSinglePoint, nucPos)
    density = np.zeros((nbas,), dtype=complex)

    #Jac = np.zeros((Tc.shape[0],))    
    #for i in range(Tc.shape[0]):
    #    Jac[i] = np.linalg.det(Jacfun(Tx, Ty, Tz))

    basex, basey, basez = cell.get_Gv_weights(cell.mesh)[1]
    b = cell.reciprocal_vectors()
    rb = np.dot(Tc, b.T)
    SIx = np.exp(-1j*np.einsum('z,g->zg', rb[:,0], basex))
    SIy = np.exp(-1j*np.einsum('z,g->zg', rb[:,1], basey))
    SIz = np.exp(-1j*np.einsum('z,g->zg', rb[:,2], basez))
    SI = SIx[:,:,None,None] * SIy[:,None,:,None] * SIz[:,None,None,:]
    natm = Tc.shape[0]
    SI = SI.reshape(natm, -1)

    #density =  -np.einsum('z,zr->r', Jac**0.5*cell.atom_charges(), SI)/cell.vol 
    #dr = np.fft.ifftn(density.reshape(cell.mesh)).flatten()/Jac**0.5 * nbas #**0.5

    density =  -np.einsum('z,zr->r', cell.atom_charges(), SI)/cell.vol 
    dr = np.fft.ifftn(density.reshape(cell.mesh)).flatten() * nbas #**0.5
    return dr

    
def getNuclearPotDenseGrid(denseMesh, nbas, nucPos, Z, invflow, flowJac, cell, productGrid = False, distortedGrid = None, JacAll=None):
    A = cell.lattice_vectors()
    a1,b1,a2,b2,a3,b3 = 0.,A[0,0],0.,A[1,1],0.,A[2,2]
    grid = cell.get_uniform_grids(denseMesh)
    if (not productGrid):
        distortedGrid = invflow(grid)
    else:
        distortedGrid = invflow(denseMesh)

    if (not productGrid):
        JacAll = flowJac(distortedGrid+1.e-6)
    else:
        JacAll = flowJac(denseMesh)

    Jac = JacAll[:,:3]     #J
    JacDet = np.asarray([np.linalg.det(Ji) for Ji in Jac]) 
    
    nmesh = cell.mesh
    cell.mesh = denseMesh
    SI = cell.get_SI()
    Gv = cell.get_Gv(denseMesh)
    vpplocG = pyscf.pbc.gto.pseudo.get_vlocG(cell, Gv)
    vpplocG = -numpy.einsum('ij,ij->j', SI, vpplocG)

    gridfft = np.array(distortedGrid*2*np.pi/A[0,0], dtype='float64')
    nucPot1 = finufft.nufft3d2(1.*gridfft[:,0], 1.*gridfft[:,1], 1.*gridfft[:,2], vpplocG.reshape(denseMesh),modeord=1,isign=1).real


    nucPot = np.zeros((nbas,))
    for i in range(nbas):
        orbval = np.zeros((nbas,), dtype=complex)
        orbval[i] = 1.
        orbDense = np.fft.ifftn(FFTInterpolation.PadZeroForInterpolation3d(np.fft.fftn(orbval.reshape(nmesh)), nmesh, denseMesh)).flatten()
        nucPot[i] = orbDense.conj().dot(nucPot1/JacDet**0.5).real/cell.vol

    cell.mesh = nmesh
    return nucPot
    
def getNuclearPotDenseGridNoDiagonal(denseMesh, nbas, nucPos, Z, invflow, flowJac, cell, productGrid = False, distortedGrid = None, JacAll=None):
    A = cell.lattice_vectors()
    a1,b1,a2,b2,a3,b3 = 0.,A[0,0],0.,A[1,1],0.,A[2,2]
    grid = cell.get_uniform_grids(denseMesh)
    if (not productGrid):
        distortedGrid = invflow(grid)
    else:
        distortedGrid = invflow(denseMesh)

    if (not productGrid):
        JacAll = flowJac(distortedGrid+1.e-6)
    else:
        JacAll = flowJac(denseMesh)

    Jac = JacAll[:,:3]     #J
    JacDet = np.asarray([np.linalg.det(Ji) for Ji in Jac]) 
    
    nmesh = cell.mesh
    cell.mesh = denseMesh
    SI = cell.get_SI()
    Gv = cell.get_Gv(denseMesh)
    vpplocG = pyscf.pbc.gto.pseudo.get_vlocG(cell, Gv)
    vpplocG = -numpy.einsum('ij,ij->j', SI, vpplocG)

    gridfft = np.array(distortedGrid*2*np.pi/A[0,0], dtype='float64')
    nucPot1 = finufft.nufft3d2(1.*gridfft[:,0], 1.*gridfft[:,1], 1.*gridfft[:,2], vpplocG.reshape(denseMesh),modeord=1,isign=1).real


    orbVal = np.zeros((nbas,np.prod(denseMesh)), dtype=complex)

    for i in range(nbas):
        orbvali = np.zeros((nbas,), dtype=complex)
        orbvali[i] = 1.
        orbVal[i] = np.fft.ifftn(FFTInterpolation.PadZeroForInterpolation3d(np.fft.fftn(orbvali.reshape(nmesh)), nmesh, denseMesh)).flatten() * (np.prod(denseMesh)/np.prod(nmesh))**0.5

    ##nuclear operator
    nucPot = orbVal.conj().dot(np.einsum('ar,r->ar', orbVal, nucPot1).T)
    nucPot = nucPot.real/cell.vol


    ##overlap matrix
    S = orbVal.conj().dot(orbVal.T)

    cell.mesh = nmesh
    return nucPot
    
def getNuclearKineticDenseGrid(denseMesh, nbas, nucPos, Z, invflow, flowJac, cell, productGrid = False, distortedGrid = None, JacAll=None):
    A = cell.lattice_vectors()
    a1,b1,a2,b2,a3,b3 = 0.,A[0,0],0.,A[1,1],0.,A[2,2]
    grid = cell.get_uniform_grids(denseMesh)
    if (not productGrid):
        distortedGrid = invflow(grid)
    else:
        distortedGrid = invflow(denseMesh)

    if (not productGrid):
        JacAll = flowJac(distortedGrid+1.e-6)
    else:
        JacAll = flowJac(denseMesh)

    Jac = JacAll[:,:3]     #J
    JacDet = np.asarray([np.linalg.det(Ji) for Ji in Jac]) 
    
    nmesh = cell.mesh
    cell.mesh = denseMesh
    SI = cell.get_SI()
    Gv = cell.get_Gv(denseMesh)
    vpplocG = pyscf.pbc.gto.pseudo.get_vlocG(cell, Gv)
    vpplocG = -numpy.einsum('ij,ij->j', SI, vpplocG)

    gridfft = np.array(distortedGrid*2*np.pi/A[0,0], dtype='float64')
    nucPot1 = finufft.nufft3d2(1.*gridfft[:,0], 1.*gridfft[:,1], 1.*gridfft[:,2], vpplocG.reshape(denseMesh),modeord=1,isign=1).real


    basis = np.zeros((nbas, np.prod(denseMesh)), dtype=complex)
    for i in range(nbas):
        orbval = np.zeros((nbas,), dtype=complex)
        orbval[i] = 1.

        basis[i] = np.fft.ifftn(FFTInterpolation.PadZeroForInterpolation3d(np.fft.fftn(orbval.reshape(nmesh)), nmesh, denseMesh)).flatten() * (np.prod(denseMesh)/nbas)**0.5


    #temp = np.einsum('bx,x->bx', basis, nucPot1)/cell.vol
    #Nuc = np.einsum('ax,bx->ab', basis.conj(), temp) / cell.vol

    return basis, nucPot1 #temp
    
    
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

    L = cell.lattice_vectors()[0,0]
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


    grid = grid * 2 * np.pi/L  ###this should be different
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
def Condition(v, G2, basMesh, e0=0):

    v0 = np.fft.fftn(v.reshape(basMesh)).flatten()
    v0 = v0/(G2-e0)
    v0[0] = 0.
    v0 = np.fft.ifftn(v0.reshape(basMesh)).flatten()

    return v0.real

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

    return Hv.real

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

def Ex(orb, mo, V2e, dw):
    nmo = mo.shape[0]  
    orbOut = 0.*orb
    for i in range(nmo):
        den = mo[i]*orb
        Ji = V2e(den, 0.*den) * dw
        orbOut += mo[i]*Ji
    return orbOut

def ExACE(orb, ACEmo):
    nmo = ACEmo.shape[0]
    orbOut = 0.*orb

    w = ACEmo.dot(orb) #np.einsum('ig,g->i', mo, orb)
    return ACEmo.T.dot(w)

def makeACE(mo, V2e, dw):
    nmo = mo.shape[0]
    M = np.zeros((nmo, nmo))
    W = 0.*mo

    for i in range(nmo):
        W[i] = Ex(mo[i], mo, V2e, dw)

    M = mo.dot(W.T)
    Minv = np.linalg.inv(M)
    L = np.linalg.cholesky(Minv)
    return L.T.dot(W)

@jit
def gaussVal(alpha, m, center, grid, A, B, C, norm):
    val =  (center[0]-grid[:,0]+0)**m[0]*(center[1]-grid[:,1]+0)**m[1]*(center[2]-grid[:,2]+0)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0])**2) * jnp.exp(-alpha*(center[1]-grid[:,1])**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+0)**2)
    val += (center[0]-grid[:,0]+0)**m[0]*(center[1]-grid[:,1]+0)**m[1]*(center[2]-grid[:,2]+C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0])**2) * jnp.exp(-alpha*(center[1]-grid[:,1])**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+C)**2)
    val += (center[0]-grid[:,0]+0)**m[0]*(center[1]-grid[:,1]+0)**m[1]*(center[2]-grid[:,2]-C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0])**2) * jnp.exp(-alpha*(center[1]-grid[:,1])**2) * jnp.exp(-alpha*(center[2]-grid[:,2]-C)**2)

    val += (center[0]-grid[:,0]+A)**m[0]*(center[1]-grid[:,1]+0)**m[1]*(center[2]-grid[:,2]+0)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]+A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1])**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+0)**2)
    val += (center[0]-grid[:,0]+A)**m[0]*(center[1]-grid[:,1]+0)**m[1]*(center[2]-grid[:,2]+C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]+A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1])**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+C)**2)
    val += (center[0]-grid[:,0]+A)**m[0]*(center[1]-grid[:,1]+0)**m[1]*(center[2]-grid[:,2]-C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]+A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1])**2) * jnp.exp(-alpha*(center[2]-grid[:,2]-C)**2)

    val += (center[0]-grid[:,0]-A)**m[0]*(center[1]-grid[:,1]+0)**m[1]*(center[2]-grid[:,2]+0)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]-A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1])**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+0)**2)
    val += (center[0]-grid[:,0]-A)**m[0]*(center[1]-grid[:,1]+0)**m[1]*(center[2]-grid[:,2]+C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]-A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1])**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+C)**2)
    val += (center[0]-grid[:,0]-A)**m[0]*(center[1]-grid[:,1]+0)**m[1]*(center[2]-grid[:,2]-C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]-A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1])**2) * jnp.exp(-alpha*(center[2]-grid[:,2]-C)**2)

    val += (center[0]-grid[:,0]+0)**m[0]*(center[1]-grid[:,1]+B)**m[1]*(center[2]-grid[:,2]+0)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0])**2) * jnp.exp(-alpha*(center[1]-grid[:,1]+B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+0)**2)
    val += (center[0]-grid[:,0]+0)**m[0]*(center[1]-grid[:,1]+B)**m[1]*(center[2]-grid[:,2]+C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0])**2) * jnp.exp(-alpha*(center[1]-grid[:,1]+B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+C)**2)
    val += (center[0]-grid[:,0]+0)**m[0]*(center[1]-grid[:,1]+B)**m[1]*(center[2]-grid[:,2]-C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0])**2) * jnp.exp(-alpha*(center[1]-grid[:,1]+B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]-C)**2)

    val += (center[0]-grid[:,0]+A)**m[0]*(center[1]-grid[:,1]+B)**m[1]*(center[2]-grid[:,2]+0)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]+A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1]+B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+0)**2)
    val += (center[0]-grid[:,0]+A)**m[0]*(center[1]-grid[:,1]+B)**m[1]*(center[2]-grid[:,2]+C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]+A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1]+B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+C)**2)
    val += (center[0]-grid[:,0]+A)**m[0]*(center[1]-grid[:,1]+B)**m[1]*(center[2]-grid[:,2]-C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]+A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1]+B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]-C)**2)

    val += (center[0]-grid[:,0]-A)**m[0]*(center[1]-grid[:,1]+B)**m[1]*(center[2]-grid[:,2]+0)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]-A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1]+B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+0)**2)
    val += (center[0]-grid[:,0]-A)**m[0]*(center[1]-grid[:,1]+B)**m[1]*(center[2]-grid[:,2]+C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]-A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1]+B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+C)**2)
    val += (center[0]-grid[:,0]-A)**m[0]*(center[1]-grid[:,1]+B)**m[1]*(center[2]-grid[:,2]-C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]-A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1]+B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]-C)**2)

    val += (center[0]-grid[:,0]+0)**m[0]*(center[1]-grid[:,1]-B)**m[1]*(center[2]-grid[:,2]+0)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0])**2) * jnp.exp(-alpha*(center[1]-grid[:,1]-B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+0)**2)
    val += (center[0]-grid[:,0]+0)**m[0]*(center[1]-grid[:,1]-B)**m[1]*(center[2]-grid[:,2]+C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0])**2) * jnp.exp(-alpha*(center[1]-grid[:,1]-B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+C)**2)
    val += (center[0]-grid[:,0]+0)**m[0]*(center[1]-grid[:,1]-B)**m[1]*(center[2]-grid[:,2]-C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0])**2) * jnp.exp(-alpha*(center[1]-grid[:,1]-B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]-C)**2)

    val += (center[0]-grid[:,0]+A)**m[0]*(center[1]-grid[:,1]-B)**m[1]*(center[2]-grid[:,2]+0)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]+A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1]-B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+0)**2)
    val += (center[0]-grid[:,0]+A)**m[0]*(center[1]-grid[:,1]-B)**m[1]*(center[2]-grid[:,2]+C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]+A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1]-B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+C)**2)
    val += (center[0]-grid[:,0]+A)**m[0]*(center[1]-grid[:,1]-B)**m[1]*(center[2]-grid[:,2]-C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]+A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1]-B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]-C)**2)

    val += (center[0]-grid[:,0]-A)**m[0]*(center[1]-grid[:,1]-B)**m[1]*(center[2]-grid[:,2]+0)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]-A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1]-B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+0)**2)
    val += (center[0]-grid[:,0]-A)**m[0]*(center[1]-grid[:,1]-B)**m[1]*(center[2]-grid[:,2]+C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]-A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1]-B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]+C)**2)
    val += (center[0]-grid[:,0]-A)**m[0]*(center[1]-grid[:,1]-B)**m[1]*(center[2]-grid[:,2]-C)**m[2]*jnp.exp(-alpha*(center[0]-grid[:,0]-A)**2) * jnp.exp(-alpha*(center[1]-grid[:,1]-B)**2) * jnp.exp(-alpha*(center[2]-grid[:,2]-C)**2)

    return val*norm*(-1.)**(m[0]+m[1]+m[2])


@jit
def ffun(x,a,b,n):
    nx = x.shape[0]
    u = jnp.reshape( 2*jnp.pi*(x - (a+b)/2)/(b-a), (nx,1) )
    f = jnp.exp(1j*n*u) / jnp.sqrt(b-a)
    return f

# define Fourier mode derivative evaluator (not properly normalized...)
@jit
def dffun(x,a,b,n):
    nx = x.shape[0]
    u = jnp.reshape( 2*jnp.pi*(x - (a+b)/2)/(b-a), (nx,1) )
    df = ((2*jnp.pi)/(b-a)) * 1j*n * jnp.exp(1j*n*u) / jnp.sqrt(b-a)
    return df

@jit
def fourier_fastrak(C_T1,C_T2,C_T3,xp1,xp2,xp3,a1,b1,a2,b2,a3,b3,nx,ny,nz):

    np1 = xp1.shape[0]
    np2 = xp2.shape[0]
    np3 = xp3.shape[0]

    fp1 = ffun(xp1,a1,b1,nx)
    fp2 = ffun(xp2,a2,b2,ny)
    fp3 = ffun(xp3,a3,b3,nz)

    T1fit = jnp.reshape(xp1,(np1,1,1)) + jnp.real ( jnp.einsum('abc,xa,yb,zc->xyz',C_T1,fp1,fp2,fp3) )
    T2fit = jnp.reshape(xp2,(1,np2,1)) + jnp.real ( jnp.einsum('abc,xa,yb,zc->xyz',C_T2,fp1,fp2,fp3) )
    T3fit = jnp.reshape(xp3,(1,1,np3)) + jnp.real ( jnp.einsum('abc,xa,yb,zc->xyz',C_T3,fp1,fp2,fp3) )

    return T1fit,T2fit,T3fit


@jit
def fourier_fastrak_jac(C_T1,C_T2,C_T3,xp1,xp2,xp3,a1,b1,a2,b2,a3,b3, nx, ny, nz):

    np1 = xp1.shape[0]
    np2 = xp2.shape[0]
    np3 = xp3.shape[0]

    fp1 = ffun(xp1,a1,b1,nx)
    fp2 = ffun(xp2,a2,b2,ny)
    fp3 = ffun(xp3,a3,b3,nz)

    dfp1 = dffun(xp1,a1,b1,nx)
    dfp2 = dffun(xp2,a2,b2,ny)
    dfp3 = dffun(xp3,a3,b3,nz)


    J11fit = 1.0 + jnp.real( jnp.einsum('abc,xa,yb,zc->xyz',C_T1,dfp1,fp2,fp3) )
    J12fit = jnp.real( jnp.einsum('abc,xa,yb,zc->xyz',C_T1,fp1,dfp2,fp3) )
    J13fit = jnp.real( jnp.einsum('abc,xa,yb,zc->xyz',C_T1,fp1,fp2,dfp3) )

    J21fit = jnp.real( jnp.einsum('abc,xa,yb,zc->xyz',C_T2,dfp1,fp2,fp3) )
    J22fit = 1.0 + jnp.real( jnp.einsum('abc,xa,yb,zc->xyz',C_T2,fp1,dfp2,fp3) )
    J23fit = jnp.real( jnp.einsum('abc,xa,yb,zc->xyz',C_T2,fp1,fp2,dfp3) )

    J31fit = jnp.real( jnp.einsum('abc,xa,yb,zc->xyz',C_T3,dfp1,fp2,fp3) )
    J32fit = jnp.real( jnp.einsum('abc,xa,yb,zc->xyz',C_T3,fp1,dfp2,fp3) )
    J33fit = 1.0 + jnp.real( jnp.einsum('abc,xa,yb,zc->xyz',C_T3,fp1,fp2,dfp3) )

    DT = jnp.zeros((np1,np2,np3,3,3))
    DT = DT.at[:,:,:,0,0].set(J11fit)
    DT = DT.at[:,:,:,0,1].set(J12fit)
    DT = DT.at[:,:,:,0,2].set(J13fit)
    DT = DT.at[:,:,:,1,0].set(J21fit)
    DT = DT.at[:,:,:,1,1].set(J22fit)
    DT = DT.at[:,:,:,1,2].set(J23fit)
    DT = DT.at[:,:,:,2,0].set(J31fit)
    DT = DT.at[:,:,:,2,1].set(J32fit)
    DT = DT.at[:,:,:,2,2].set(J33fit)

    J = jnp.linalg.det(DT)

    return DT,J

#'''
def optimizeFourier(L, nmesh, Fx, Fy, Fz, nx, ny, nz, gx, gy, gz, alpha, m, center, norm, Smat, dw):
    A, B, C = L[0,0], L[1,1], L[2,2]
    
    Tbasx, Tbasy, Tbasz = fourier_fastrak(Fx, Fy, Fz, gx, gy, gz, 0, A, 0, B, 0, C, nx, ny, nz)
    Tbasx, Tbasy, Tbasz = Tbasx[:-1,:-1,:-1], Tbasy[:-1,:-1,:-1], Tbasz[:-1,:-1,:-1]
    grid = jnp.hstack((Tbasx.reshape(-1,1), Tbasy.reshape(-1,1), Tbasz.reshape(-1,1))).real

    DT, J = fourier_fastrak_jac(Fx, Fy, Fz, gx, gy, gz, 0, A, 0, B, 0, C, nx, ny, nz)
    DT = DT[:-1,:-1,:-1]
    DT = DT.reshape(-1,3,3)
    JacDet = jnp.asarray([jnp.linalg.det(Ji) for Ji in DT])

    aoVals =  vmap(gaussVal, in_axes = (0, 0, 0, None, None, None, None, 0))(alpha, m, center, grid, A, B, C, norm) 
    S = jnp.einsum('ag,g,bg->ab', aoVals, JacDet, aoVals)
    #return S
    #print(ao2.shape)
    #S = jnp.einsum('ag,bg->ab', ao2, aoVals) * dw
    #return S
    #print(S.shape, Smat.shape)
    error = jnp.linalg.norm(S-Smat)
    #print(error)
    return error
            
def HF(cell, basMesh, nmesh, mf, invFlow, flow, Jacfun, ACE = True,  eps = 1.e-6):
    A = cell.lattice_vectors()
    a1,b1,a2,b2,a3,b3 = 0.,A[0,0],0.,A[1,1],0.,A[2,2]

    grid = cell.get_uniform_grids(nmesh)
    basgrid = cell.get_uniform_grids(basMesh)
    
    '''
    dx, dy, dz, _ = invFlow(grid[:,0], grid[:,1], grid[:,2])
    Jac = Jacfun(dx, dy, dz)
    Jac = np.asarray([np.linalg.inv(Ji) for Ji in Jac])
    distortedGrid = jnp.hstack((dx, dy, dz))
    JacDet = np.asarray([np.linalg.det(Ji) for Ji in Jac])  #|J|
    '''

    dx, dy, dz, _ = invFlow(grid[:,0], grid[:,1], grid[:,2])
    Jac = Jacfun(grid[:,0], grid[:,1], grid[:,2])
    Jac = np.asarray([np.linalg.inv(Ji) for Ji in Jac])
    JacDet = np.asarray([np.linalg.det(Ji) for Ji in Jac])  #|J|
    distortedGrid = jnp.hstack((dx.reshape(-1,1), dy.reshape(-1,1), dz.reshape(-1,1)))
    
    ngrid, nbas = np.prod(nmesh), np.prod(basMesh)
    G = get_Gv(nmesh, cell)
    G2 = np.einsum('gi,gi->g', G, G)

    t0 = time.time()

        

    
    from scipy.sparse.linalg import LinearOperator    
    G2[0] = 1.
    Hvop2   = LinearOperator((nbas, nbas), matvec = lambda v: Hv2(v, JacDet, Jac, basMesh, G))        
    preCond = LinearOperator((nbas, nbas), matvec = lambda v: Condition(v, G2, basMesh))
    printNorm = lambda v : print(np.linalg.norm(v))

    v0 = 1./JacDet**0.5
    v0 = v0/np.linalg.norm(v0)

    
    nucPos = np.asarray([cell._env[cell._atm[i,1]:cell._atm[i,1]+3] for i in range(cell._atm.shape[0])])


    v0 = 1./JacDet**0.5
    v0 = v0/np.linalg.norm(v0)
    def V2e(vec, guess, tol=1.e-12):
        vec2 = vec*JacDet**0.5
        
        vec2 = vec2 - vec2.dot(v0) * v0
        #pot, info = scipy.sparse.linalg.cg(Hvop2, vec2, x0=guess, M=preCond, callback=printNorm, tol = tol**0.5/np.linalg.norm(vec2), atol=tol**0.5)
        pot, info = scipy.sparse.linalg.cg(Hvop2, vec2, x0=guess, M=preCond, tol = tol**0.5/np.linalg.norm(vec2), atol=tol**0.5)
        if (info > 0):
            print("conjugate gradient convergence not achieved!!!!!")
            exit(0)

        potout = pot - pot.dot(v0) * v0

        #error = Hv2(potout, JacDet, Jac, basMesh, G) - vec2
        #print("final error ", np.linalg.norm(error))
        
        potout = potout * JacDet**0.5
        return potout.real * 4. * np.pi 
        

    #denseMesh = nmesh #[46,46,46]
    #nucPot = getNuclearPotDenseGrid(denseMesh, nbas, nucPos, cell._atm[:,0], \
    #    invFlow, vmap(Tdel, in_axes=(0)), cell, distortedGrid, JacAll) * JacDet**0.5

    if cell.pseudo:
        nGnuc = 60
        denseMesh = [max(nGnuc,nmesh[0]),max(nGnuc,nmesh[0]),max(nGnuc,nmesh[0])]#[12,12,12]
        nucPot = getNuclearPotDenseGrid(denseMesh, nbas, nucPos, cell._atm[:,0], \
            invFlow, JacAllFun, cell, productGrid) * JacDet**0.5
    else:
        nucDensity = getStructureFactor(nbas, flow, Jacfun, invFlow, cell).real
        nucPot = V2e(nucDensity, 0.*nucDensity).real #* nbas/cell.vol
    #nucPot = getNuclearPotDenseGridNoDiagonal(denseMesh, nbas, nucPos, cell._atm[:,0], \
    #    invFlow, JacAllFun, cell, productGrid) 
        

    #'''
    N1 = mf.with_df.get_pp()
    S1 = cell.pbc_intor('cint1e_ovlp_sph')
    K1 = cell.pbc_intor('cint1e_kin_sph')
    ao = cell.pbc_eval_gto('GTOval', distortedGrid)

    S = jnp.einsum('ai,a,aj->ij', ao, 1./JacDet, ao) * cell.vol/nbas
    N = jnp.einsum('ai,a,aj->ij', ao, nucPot/JacDet, ao) * cell.vol/nbas

    K = 0.*K1
    for i in range(ao.shape[1]):
        kao = Hv2(ao[:,i]/JacDet**0.5, JacDet, Jac, basMesh, G).real/2.
        K[i] = jnp.einsum('g,gi->i', kao/JacDet**0.5, ao) * cell.vol/nbas 
         
    print(" nuc  ", (N-N1).max())  
    print(" kin  ", (K-K1).max(), np.unravel_index(abs(K-K1).argmax(), K.shape))  
    print(" ovlp ", (S-S1).max())  

    nocc = cell.nelectron//2
    mocoeff = mf.mo_coeff
    if isinstance(mocoeff, list):
        mocoeff = mocoeff[0]
        
    dm = 2*mocoeff[:,:nocc].dot(mocoeff[:,:nocc].T)
    jao, kao = mf.get_jk(dm)
    jmo, kmo = jnp.einsum('ik,ij,jl->kl', mocoeff[:,:nocc], jao, mocoeff[:,:nocc]), np.einsum('ik,ij,jl->kl', mocoeff[:,:nocc], kao, mocoeff[:,:nocc])
    nuc, kin = jnp.einsum('ik,ij,jl->kl', mocoeff[:,:nocc], N1, mocoeff[:,:nocc]), np.einsum('ik,ij,jl->kl', mocoeff[:,:nocc], K1, mocoeff[:,:nocc])
    moVal = np.einsum('gi,g->ig', ao.dot(mocoeff[:,:nocc]), 1./JacDet**0.5)
    density = 2.*np.einsum('ig,ig->g', moVal, moVal)
    J = V2e(density, 0*density)
    coul = (J*density/2).sum()*cell.vol/nbas
    


    #'''

    #i = 0
    #def hv(v):
    #    return nucPot*v + Hv2(v, JacDet, Jac, basMesh, G)/2.
    #Ham   = LinearOperator((nbas, nbas), matvec = hv )        
    #(w,v) = eigsh(Ham, 1, which = 'SA')
    
    #print(w[0], cell.energy_nuc())
    
    nelec, nuclearPotential = cell.nelectron, cell.energy_nuc()
    
    def FC(C, J, orbs=None, ACE=False):
        C = np.asarray(C)
        Cout = 0.*np.asarray(C)
        if (len(C.shape) == 2 ):
            for i in range(C.shape[0]):
                Cout[i] = (J+nucPot)*C[i] + Hv2(C[i], JacDet, Jac, basMesh, G).real/2. 
                if (orbs is not None and ACE is False):
                    Cout[i] -= Ex(C[i], orbs, V2e, nbas/cell.vol)
                elif (orbs is not None and ACE is True):
                    Cout[i] -= ExACE(C[i], orbs)
                #Cout[i] = (J)*C[i] + nucPot.dot(C[i]) + Hv2(C[i], JacDet, Jac, basMesh, G).real/2. 
            return Cout

        else :
            if (orbs is not None):
                if (not ACE):
                    return (J+nucPot)*C + Hv2(C, JacDet, Jac, basMesh, G).real/2. - Ex(C[i], orbs, V2e, nbas/cell.vol)
                else:
                    return (J+nucPot)*C + Hv2(C, JacDet, Jac, basMesh, G).real/2. - ExACE(C[i], orbs)
            return (J+nucPot)*C + Hv2(C, JacDet, Jac, basMesh, G).real/2.         
    
    def precond(x, e0):
        return Condition(x, G2/2, basMesh, e0)
    
    if (mf.mo_coeff is not None):
        orbs = np.einsum('ij,j->ij', mf.mo_coeff[:,:cell.nelectron//2].T.dot(ao.T), 1/JacDet**0.5) * (cell.vol/nbas)**0.5

    else:
        Ck = np.random.random((cell.nelectron//2,nbas))
        conv, e, orbs = Davidson.davidson1(lambda C : FC(C, 0.*G2), Ck, precond, nroots=nelec//2, verbose=0, tol=1e-3)
        orbs = np.asarray(orbs)
    
    #conv, e, x0 = Davidson.davidson1(lambda C : FC(C, 0.*G2), Ck, precond, nroots=nelec//2, verbose=0)
    #print(basMesh, conv, e)

    aceOrbs = makeACE(orbs, V2e, nbas/cell.vol)    
    density = 2*np.einsum('ig,ig->g', orbs, orbs.conj())
    j = V2e(density.real, 0.*G2) * nbas/cell.vol
    #Ho = FC(orbs, j, aceOrbs, True)
    Ht, Hn, Hj, Hk = 0.*orbs, 0.*orbs, 0.*orbs, 0.*orbs


    for i in range(orbs.shape[0]):
        Ht[i] = Hv2(orbs[i], JacDet, Jac, basMesh, G).real/2. 
        Hn[i] = (nucPot)*orbs[i]
        Hj[i] = J*orbs[i]
        Hk[i] = ExACE(orbs[i], aceOrbs)
        t, n, j, k = orbs[i].dot(Ht[i].T), orbs[i].dot(Hn[i].T), orbs[i].dot(Hj[i].T), orbs[i].dot(Hk[i].T)
        print("{0:3d}  {1:14.5f}  {2:14.5f}  {3:14.5f}  {4:14.5f} -> {5:14.5f}".format(i, t, n, j, k, t+n+j-k))
        print("{0:3d}  {1:14.5f}  {2:14.5f}  {3:14.5f}  {4:14.5f} -> {5:14.5f}".format(i, kin[i,i], nuc[i,i], jmo[i,i], kmo[i,i]/2., kin[i,i]+nuc[i,i]+jmo[i,i]-kmo[i,i]/2))
        print("{0:3d}  {1:14.5f}  {2:14.5f}  {3:14.5f}  {4:14.5f} (ovlp) -> {5:14.5f}".format(i, kin[i,i]-t, nuc[i,i]-n, jmo[i,i]-j, kmo[i,i]/2.-k, orbs[i].dot(orbs[i])-1.))
        print()

    print("Kin  Error {0:14.8f}".format(2*np.einsum('ig,ig', orbs, Ht) - 2*kin.diagonal().sum()))
    print("Nuc  Error {0:14.8f}".format(2*np.einsum('ig,ig', orbs, Hn) - 2*nuc.diagonal().sum()))
    print("Coul Error {0:14.8f}".format( np.einsum('ig,ig', orbs, Hj) - jmo.diagonal().sum()))
    print("Exc  Error {0:14.8f}".format( np.einsum('ig,ig', orbs, Hk) - kmo.diagonal().sum()/2.))
    Energy = 2*np.einsum('ig,ig', orbs, Ht) + 2*np.einsum('ig,ig', orbs, Hn) + np.einsum('ig,ig', orbs, Hj) - np.einsum('ig,ig', orbs, Hk) + nuclearPotential    
    print("{0:14.8f}  {1:14.8f} {2:14.8f}".format(Energy, mf.e_tot, Energy-mf.e_tot))
    BetterEnergy = 2*np.einsum('ag,ab,bg', mocoeff[:,:nocc], K1, mocoeff[:,:nocc]) + 2*np.einsum('ag,ab,bg', mocoeff[:,:nocc], N1, mocoeff[:,:nocc])  + np.einsum('ig,ig', orbs, Hj) - np.einsum('ig,ig', orbs, Hk) + nuclearPotential 
    print("BestGuess (distroted Sinc DF) {0:14.8f}".format(BetterEnergy))

    #orbs = moVal*(cell.vol/nbas)**0.5
    #orbs = np.einsum('gi,ij,g->jg', ao, mocoeff[:,:1], 1./JacDet**0.5) * (cell.vol/nbas)**0.5 #(ao.dot(mocoeff[:,:1])/JacDet**0.5).T
    #Ho = FC(orbs, j, orbs) if not ACE else FC(orbs, j, aceOrbs, True)
    #Energy = 2.*np.einsum('ig,ig', orbs, Ho)  + nuclearPotential


    Energy = 0.
    de = 1.e-6
    t0 = time.time()
    charge_convtol, convtol, de = 0.1, 1.e-6, 1.

    for outer in range(20):
        #dc = FDiisContext(10)

        if (ACE):
            aceOrbs = makeACE(orbs, V2e, nbas/cell.vol)
            
        density = 2*np.einsum('ig,ig->g', orbs, orbs.conj())
        j = V2e(density.real, 0.*G2) * nbas/cell.vol
        
        outerEold = Energy
        from pyscf.lib.diis import DIIS    
        dc = DIIS()
        dc.space = 10
        
        if (outer > 0):
            charge_convtol = min( charge_convtol, max(convtol, 0.1*abs(de)))
        davidsonTol = max(convtol*0.1, charge_convtol*0.01)

        #print(davidsonTol, convtol, charge_convtol)
        for it in range(8):
            conv, e, orbs = Davidson.davidson1(lambda C : FC(C, j, orbs) if not ACE else FC(C, j, aceOrbs, True), #FC(C, j, orbs), 
                                            orbs, precond, nroots=nelec//2, verbose=0, tol=davidsonTol)
            #orbs, e, iter, maxerror = Davidson.Davidson(lambda C : FC(C, j), precond, Ck)
            orbs = np.asarray(orbs)
            
            density = 2*np.einsum('ig,ig->g', orbs, orbs.conj())
            oldj = 1.*j
            j = V2e(density, j/(nbas/cell.vol)/4./np.pi) * nbas/cell.vol
            
            error = j - oldj
            
            oldE = Energy
            Ho = FC(orbs, j, orbs) if not ACE else FC(orbs, j, aceOrbs, True)
            Ho1 = FC(orbs, 0*j)
            Energy = np.einsum('ig,ig', orbs, Ho) + np.einsum('ig,ig', orbs, Ho1)  + nuclearPotential
            errorNorm = np.linalg.norm(error/JacDet**0.5) * (cell.vol/nbas)**0.5

            de = abs(Energy - oldE).real
            davidsonTol = max(1.e-10, min(davidsonTol, 0.001*de))

            dt = time.time() - t0
            if (de< charge_convtol and False):

                break

            #j = dc.Apply(j, error)[0]
            j = dc.update(j, error)
        #'''
        
        print("{0:3d}   {1:15.8f}  {2:15.8f}  {3:15.8f}  {4:15.2f}".format(outer, Energy.real, errorNorm, (Energy - outerEold), dt) )
        #print()
        if (abs(Energy - outerEold) < charge_convtol):
            break
               
        if (ACE):
            aceOrbs = makeACE(orbs, V2e, nbas/cell.vol)

        

    return Energy.real , orbs 


def get_transformation_matrix(S, lin_dep_thresh):
    #Diagonalize the overlap matrix S to obtain the transformation matrix X
    Seigval,Sorbs = numpy.linalg.eigh(S)
    emin = Seigval[-1]*lin_dep_thresh
    X = Sorbs[:,Seigval>emin]
    w = Seigval[Seigval>emin]

    numpy.sqrt(w, out=w)
    w[:] = 1./w
    X = X*w
    return X

def getOrbs(X, Fock):

    Fockk = X.conj().T.dot(Fock.dot(X))
    e, orbsk = numpy.linalg.eigh(Fockk)
    e = e.real
    mo = X.dot(orbsk) # Don't use this for anything rn.

    return mo, e

def occRI(moorbs,S,Koa,Koo):
    Sa = np.dot(S, moorbs.T)
    tmp = np.dot(Sa, Koa)
    Kuv = tmp
    Kuv += tmp.T 
    np.dot(Koo, Sa.T, Koa)
    Kuv -= np.dot(Sa, Koa)
    return Kuv

def getE(Hx, dm, CoreH, nuc):
    energy = numpy.einsum('ij,ij',Hx+CoreH, dm)
    energy += nuc
    return energy

def HF_ISDF_DistrotedGrid(cell, basMesh, nmesh, mf, invFlow, flow, Jacfun, ACE = True,  eps = 1.e-6):
    A = cell.lattice_vectors()
    a1,b1,a2,b2,a3,b3 = 0.,A[0,0],0.,A[1,1],0.,A[2,2]

    grid = cell.get_uniform_grids(nmesh)
    basgrid = cell.get_uniform_grids(basMesh)
    
    dx, dy, dz, _ = invFlow(grid[:,0], grid[:,1], grid[:,2])
    Jac = Jacfun(grid[:,0], grid[:,1], grid[:,2])
    Jac = np.asarray([np.linalg.inv(Ji) for Ji in Jac])
    JacDet = np.asarray([np.linalg.det(Ji) for Ji in Jac])  #|J|
    distortedGrid = jnp.hstack((dx.reshape(-1,1), dy.reshape(-1,1), dz.reshape(-1,1)))
    
    ngrid, nbas = np.prod(nmesh), np.prod(basMesh)
    G = get_Gv(nmesh, cell)
    G2 = np.einsum('gi,gi->g', G, G)

    t0 = time.time()

        

    
    from scipy.sparse.linalg import LinearOperator    
    G2[0] = 1.
    Hvop2   = LinearOperator((nbas, nbas), matvec = lambda v: Hv2(v, JacDet, Jac, basMesh, G))        
    preCond = LinearOperator((nbas, nbas), matvec = lambda v: Condition(v, G2, basMesh))
    printNorm = lambda v : print(np.linalg.norm(v))

    v0 = 1./JacDet**0.5
    v0 = v0/np.linalg.norm(v0)

    

    v0 = 1./JacDet**0.5
    v0 = v0/np.linalg.norm(v0)
    def V2e(vec, guess, tol=1.e-12):
        vec2 = vec*JacDet**0.5
        
        vec2 = vec2 - vec2.dot(v0) * v0
        #pot, info = scipy.sparse.linalg.cg(Hvop2, vec2, x0=guess, M=preCond, callback=printNorm, tol = tol**0.5/np.linalg.norm(vec2), atol=tol**0.5)
        pot, info = scipy.sparse.linalg.cg(Hvop2, vec2, x0=guess, M=preCond, tol = tol**0.5/np.linalg.norm(vec2), atol=tol**0.5)
        if (info > 0):
            print("conjugate gradient convergence not achieved!!!!!")
            exit(0)

        potout = pot - pot.dot(v0) * v0

        #error = Hv2(potout, JacDet, Jac, basMesh, G) - vec2
        #print("final error ", np.linalg.norm(error))
        
        potout = potout * JacDet**0.5
        return potout.real * 4. * np.pi 
        

    N1 = mf.with_df.get_pp()
    S1 = cell.pbc_intor('cint1e_ovlp_sph')
    K1 = cell.pbc_intor('cint1e_kin_sph')
    ao = cell.pbc_eval_gto('GTOval', distortedGrid)
    ao = np.einsum('gi,g->ig', ao, 1/JacDet**0.5) * (cell.vol/nbas)**0.5
    X = get_transformation_matrix(S1, 1.e-12)

    nocc = cell.nelectron//2

    nelec, nuclearPotential, nao = cell.nelectron, cell.energy_nuc(), N1.shape[0]
        

    CoreH = N1+K1
    Fock = 1*CoreH
    density, j = np.zeros((ao.shape[1],)), np.zeros((ao.shape[1],))
    Energy = 0.
    de = 1.e-6
    t0 = time.time()
    charge_convtol, convtol, de = 0.1, 1.e-6, 1.
    
    from pyscf.lib.diis import DIIS    
    dc = DIIS()
    dc.space = 10

    for iter in range(20):
        #dc = FDiisContext(10)

        mo, _ = getOrbs(X, Fock)
        orbs = mo[:,:nocc].T
        dm = 2.* orbs.T.conj().dot(orbs)
        moVal = orbs.dot(ao)

        density = 2.*np.einsum('ig,ig->g', moVal, moVal)
        j = V2e(density, j/(nbas/cell.vol)/4./np.pi) * nbas/cell.vol

        #jao, kao = mf.get_jk(dm=dm)
        Jop = jnp.einsum('ig,g,jg->ij', ao, j, ao) 
        aceOrbs = makeACE(moVal, V2e, nbas/cell.vol)

        Korb = 0.*moVal 
        for i in range(orbs.shape[0]):
            Korb[i] = ExACE(moVal[i], aceOrbs)

        Kia, Kij = Korb.dot(ao.T), Korb.dot(moVal.T)
        Kab = occRI(orbs,S1,Kia,Kij)

        FockNew = Jop + CoreH - Kab
        Energy = getE(FockNew, dm*0.5, CoreH, nuclearPotential)
        error = (FockNew - Fock)
        err_norm = numpy.linalg.norm(error)

        converged = (err_norm < 1e-5*nao)
        #print("Converged: ", time.time()-t0)
        print("{0:>8d}  {1:>18.9f}  {2:>18.3e}  {3:8.2f}".format(iter, Energy, err_norm, (time.time() - t0)), flush=True)
        if converged:
            break
        # t0 = time.time()
        Fock = dc.update(FockNew, error)


    return Energy.real , orbs 


def HFCheck(cell, basMesh, nmesh, mf, invFlow, JacAllFun, productGrid = False, eps = 1.e-6):
    a1,b1,a2,b2,a3,b3 = 0.,cell.a[0,0],0.,cell.a[1,1],0.,cell.a[2,2]

    grid = cell.get_uniform_grids(nmesh)
    basgrid = cell.get_uniform_grids(basMesh)
    

    if (not productGrid):
        distortedGrid = invFlow(grid)
    else:
        distortedGrid = invFlow(nmesh)
    
    ngrid, nbas = np.prod(nmesh), np.prod(basMesh)
    G = get_Gv(nmesh, cell)
    G2 = np.einsum('gi,gi->g', G, G)

    t0 = time.time()


    if (not productGrid):
        JacAll = JacAllFun(distortedGrid)
    else:
        JacAll = JacAllFun(nmesh)
        
    Jac = JacAll[:,:3]     #J
    JacDet = np.asarray([np.linalg.det(Ji) for Ji in Jac])  #|J|

    
    from scipy.sparse.linalg import LinearOperator    
    G2[0] = 1.
    Hvop2   = LinearOperator((nbas, nbas), matvec = lambda v: Hv2(v, JacDet, Jac, basMesh, G))        
    preCond = LinearOperator((nbas, nbas), matvec = lambda v: Condition(v, G2, basMesh))
    printNorm = lambda v : print(np.linalg.norm(v))

    v0 = 1./JacDet**0.5
    v0 = v0/np.linalg.norm(v0)

    
    nucPos = np.asarray([cell._env[cell._atm[i,1]:cell._atm[i,1]+3] for i in range(cell._atm.shape[0])])


    #denseMesh = nmesh #[46,46,46]
    #nucPot = getNuclearPotDenseGrid(denseMesh, nbas, nucPos, cell._atm[:,0], \
    #    invFlow, vmap(Tdel, in_axes=(0)), cell, distortedGrid, JacAll) * JacDet**0.5

    denseMesh = [max(50,nmesh[0]),max(50,nmesh[0]),max(50,nmesh[0])]#[12,12,12]
    nucPot = getNuclearPotDenseGrid(denseMesh, nbas, nucPos, cell._atm[:,0], \
        invFlow, JacAllFun, cell, productGrid) * JacDet**0.5


    v0 = 1./JacDet**0.5
    v0 = v0/np.linalg.norm(v0)
    def V2e(vec, guess, tol=1.e-12):
        vec2 = vec*JacDet**0.5
        
        vec2 = vec2 - vec2.dot(v0) * v0
        #pot = scipy.sparse.linalg.cg(Hvop2, vec2, x0=guess, M=preCond, callback=printNorm, tol = tol**0.5/np.linalg.norm(vec2), atol=tol**0.5)
        pot = scipy.sparse.linalg.cg(Hvop2, vec2, x0=guess, M=preCond, tol = tol**0.5/np.linalg.norm(vec2), atol=tol**0.5)

        potout = pot[0] - pot[0].dot(v0) * v0

        #error = Hv2(potout, JacDet, Jac, basMesh, G) - vec2
        #print("final error ", np.linalg.norm(error))
        
        potout = potout * JacDet**0.5
        return potout.real * 4. * np.pi 
        
    import pdb
    pdb.set_trace()
    nelec, nuclearPotential = cell.nelectron, cell.energy_nuc()
    moCoeff = mf.mo_coeff[:,:nelec//2]
    ao = cell.pbc_eval_gto('GTOval', np.array(distortedGrid, dtype=np.float64))
    mo = np.einsum('ab,ra,r->br', moCoeff, ao, 1./JacDet**0.5)

    K1 = cell.pbc_intor('cint1e_kin_sph')
    S1 = cell.pbc_intor('cint1e_ovlp_sph')
    N1 = mf.with_df.get_pp()

    se = mo.conj().dot(mo.T) * cell.vol/nbas
    errorSe = se - (moCoeff.T.conj().dot(S1.dot(moCoeff)))
    print("overlap \n")
    print(se)
    print(moCoeff.T.conj().dot(S1.dot(moCoeff)))


    kmo = np.asarray([Hv2(moi, JacDet, Jac, basMesh, G)/2. for moi in mo])
    ke = kmo.conj().dot(mo.T) * cell.vol/nbas
    errorKe = ke - (moCoeff.T.conj().dot(K1.dot(moCoeff)))
    print("kinetic \n")
    print(ke)
    print(moCoeff.T.conj().dot(K1.dot(moCoeff)))
    print(ke.diagonal().sum(), (moCoeff.T.conj().dot(K1.dot(moCoeff))).diagonal().sum())

    print("nuclear\n")
    nuc =   np.einsum('ar,r,br', mo.conj(), nucPot, mo) * cell.vol/nbas
    errorNuc = nuc - (moCoeff.T.conj().dot(N1.dot(moCoeff)))
    print(nuc)
    print((moCoeff.T.conj().dot(N1.dot(moCoeff))))
    print(nuc.diagonal().sum(), (moCoeff.T.conj().dot(N1.dot(moCoeff))).diagonal().sum())


    from pyscf.pbc import df
    nao = cell.nao_nr()
    kpts = cell.make_kpts([1,1,1])
    mydf = df.DF(cell, kpts=kpts)
    
    for i, kpti in enumerate(kpts):
        for j, kptj in enumerate(kpts):
            eri_3d = []
            for LpqR, LpqI, sign in mydf.sr_loop([kpti,kptj], compact=False):
                eri_3d.append(LpqR+LpqI*1j)
            eri_3d = numpy.vstack(eri_3d).reshape(-1,nao,nao)

    
    CoulExact = np.einsum('Fab,F->ab', eri_3d, np.einsum('Fab,ab->F', eri_3d, 2*moCoeff.conj().dot(moCoeff.T)))
    JExact = moCoeff.T.conj().dot(CoulExact.dot(moCoeff))
    
    density = 2*np.einsum('ig,ig->g', mo, mo.conj())
    Coul = V2e(density, 0.*density) 
    print("Coulomb\n")
    J =   np.einsum('ar,r,br', mo.conj(), Coul, mo) * cell.vol/nbas
    print(JExact)
    print(J)
    print(J.diagonal().sum(), JExact.diagonal().sum())

    import pdb
    pdb.set_trace()
    exit(0)
    #return
    #i = 0
    #def hv(v):
    #    return nucPot*v + Hv2(v, JacDet, Jac, basMesh, G)/2.
    #Ham   = LinearOperator((nbas, nbas), matvec = hv )        
    #(w,v) = eigsh(Ham, 1, which = 'SA')
    
    #print(w[0], cell.energy_nuc())
    
    nelec, nuclearPotential = cell.nelectron, cell.energy_nuc()
    
    def FC(C, J):
        C = np.asarray(C)
        Cout = 0.*np.asarray(C)

        if (len(C.shape) == 2 ):
            for i in range(C.shape[0]):
                Cout[i] = (J+nucPot)*C[i] + Hv2(C[i], JacDet, Jac, basMesh, G).real/2. 
            return Cout

        else :
            return (J+nucPot)*C + Hv2(C, JacDet, Jac, basMesh, G).real/2.         
    
    def precond(x, e0):
        return Condition(x, G2/2, basMesh, e0)
    
    Ck = np.random.random((nelec//2,nbas))
    #conv, e, x0 = Davidson.davidson1(lambda C : FC(C, 0.*G2), Ck, precond, nroots=nelec//2, verbose=0)
    #print(basMesh, conv, e)

    conv, e, orbs = Davidson.davidson1(lambda C : FC(C, 0.*G2), Ck, precond, nroots=nelec//2, verbose=0, tol=1e-3)
    orbs = np.asarray(orbs)
    
    density = 2*np.einsum('ig,ig->g', orbs, orbs.conj())

    davidsonTol = 1.e-3 
    j = V2e(density.real, 0.*G2, davidsonTol) * nbas/cell.vol
    Ho = FC(orbs, j/2.)
    Energy = 2.*np.einsum('ig,ig', orbs, Ho)  + nuclearPotential


    #dc = FDiisContext(10)

    from pyscf.lib.diis import DIIS    
    dc = DIIS()
    dc.space = 10
    
    '''  
    ##SCF
    for it in range(40):
        conv, e, orbs = Davidson.davidson1(lambda C : FC(C, j), orbs, precond, nroots=nelec//2, verbose=1, tol=davidsonTol)
        orbs = np.asarray(orbs)
        
        oldDensity = 1.*density
        density = 2*np.einsum('ig,ig->g', orbs, orbs.conj())
        error = density - oldDensity
        
        #density = dc.Apply(density, error)[0]
        density = dc.update(density, error)
        
        oldE = Energy
        Ho = FC(orbs, j/2.)
        Energy = 2.*np.einsum('ig,ig', orbs.conj(), Ho).real  + nuclearPotential
        errorNorm = np.linalg.norm(error)
        de = abs(Energy - oldE).real
        davidsonTol = min(davidsonTol, 0.001*de)
        
        print("{0:3d}   {1:15.8f}  {2:15.8f}  {3:15.8f}".format(it, Energy.real, errorNorm, de) )
        if (errorNorm < 1.e-5 and de < 1.e-8):
            print(errorNorm, de)
            break

        j = V2e(density, j/(nbas/cell.vol)/4./np.pi, davidsonTol)* nbas/cell.vol


    ##SCF
    '''
    t0 = time.time()
    for it in range(40):
        conv, e, orbs = Davidson.davidson1(lambda C : FC(C, j), orbs, precond, nroots=nelec//2, verbose=0, tol=davidsonTol)
        #orbs, e, iter, maxerror = Davidson.Davidson(lambda C : FC(C, j), precond, Ck)
        orbs = np.asarray(orbs)
        
        density = 2*np.einsum('ig,ig->g', orbs, orbs.conj())
        oldj = 1.*j
        j = V2e(density, j/(nbas/cell.vol)/4./np.pi) * nbas/cell.vol
        
        error = j - oldj
        
        oldE = Energy
        Ho = FC(orbs, j/2.)
        Energy = 2.*np.einsum('ig,ig', orbs, Ho)  + nuclearPotential
        errorNorm = np.linalg.norm(error/JacDet**0.5) * (cell.vol/nbas)**0.5

        de = abs(Energy - oldE).real
        davidsonTol = max(1.e-10, min(davidsonTol, 0.001*de))

        dt = time.time() - t0
        print("{0:3d}   {1:15.8f}  {2:15.8f}  {3:15.8f}  {4:15.2f}".format(it, Energy.real, errorNorm, de, dt) )
        if (de< 1.e-8 ):
           break

        #j = dc.Apply(j, error)[0]
        j = dc.update(j, error)
    #'''
    return Energy.real , orbs 

##NUCLEAR ENERGY IS CALCULATED EXACTLY
def HFExact(cell, basMesh, nmesh, mf, invFlow, JacAllFun, productGrid = False, eps = 1.e-6):
    a1,b1,a2,b2,a3,b3 = 0.,cell.a[0,0],0.,cell.a[1,1],0.,cell.a[2,2]

    grid = cell.get_uniform_grids(nmesh)
    basgrid = cell.get_uniform_grids(basMesh)
    

    if (not productGrid):
        distortedGrid = invFlow(grid)
    else:
        distortedGrid = invFlow(nmesh)
    
    ngrid, nbas = np.prod(nmesh), np.prod(basMesh)
    G = get_Gv(nmesh, cell)
    G2 = np.einsum('gi,gi->g', G, G)

    t0 = time.time()


    if (not productGrid):
        JacAll = JacAllFun(distortedGrid)
    else:
        JacAll = JacAllFun(nmesh)
        
    Jac = JacAll[:,:3]     #J
    JacDet = np.asarray([np.linalg.det(Ji) for Ji in Jac])  #|J|

    #factor = np.sum(1./JacDet) * 1/nbas
    #JacDet = JacDet/factor
    #print(factor)
    from scipy.sparse.linalg import LinearOperator    
    G2[0] = 1.
    Hvop2   = LinearOperator((nbas, nbas), matvec = lambda v: Hv2(v, JacDet, Jac, basMesh, G))        
    preCond = LinearOperator((nbas, nbas), matvec = lambda v: Condition(v, G2, basMesh))
    printNorm = lambda v : print(np.linalg.norm(v))

    v0 = 1./JacDet**0.5
    v0 = v0/np.linalg.norm(v0)

    print("add ", np.sum(1./JacDet)/nbas)
    nucPos = np.asarray([cell._env[cell._atm[i,1]:cell._atm[i,1]+3] for i in range(cell._atm.shape[0])])

    #denseMesh = nmesh #[46,46,46]
    #nucPot = getNuclearPotDenseGrid(denseMesh, nbas, nucPos, cell._atm[:,0], \
    #    invFlow, vmap(Tdel, in_axes=(0)), cell, distortedGrid, JacAll) * JacDet**0.5

    denseMesh = [40,40,40] #2*np.asarray(nmesh) #[12,12,12]
    Nuc1, nucPot = getNuclearKineticDenseGrid(denseMesh, nbas, nucPos, cell._atm[:,0], \
        invFlow, JacAllFun, cell, productGrid) #* JacDet**0.5


    v0 = 1./JacDet**0.5
    v0 = v0/np.linalg.norm(v0)
    def V2e(vec, guess, tol=1.e-12):
        vec2 = vec*JacDet**0.5
        
        vec2 = vec2 - vec2.dot(v0) * v0
        #pot = scipy.sparse.linalg.cg(Hvop2, vec2, x0=guess, M=preCond, callback=printNorm, tol = tol**0.5/np.linalg.norm(vec2), atol=tol**0.5)
        pot = scipy.sparse.linalg.cg(Hvop2, vec2, x0=guess, M=preCond, tol = tol/np.linalg.norm(vec2), atol=tol)

        potout = pot[0] - pot[0].dot(v0) * v0

        #error = Hv2(potout, JacDet, Jac, basMesh, G) - vec2
        #print("final error ", np.linalg.norm(error))
        
        potout = potout * JacDet**0.5
        return potout.real * 4. * np.pi 
        

    mf = pyscf.pbc.scf.RHF(cell).density_fit(auxbasis='weigend')
    mf.kernel()
    moCoeff = mf.mo_coeff
    ao = cell.pbc_eval_gto('GTOval', np.array(distortedGrid, dtype=np.float64))
    mo = np.einsum('a,ra->r', moCoeff[:,0], ao).flatten()/JacDet**0.5

    K1 = cell.pbc_intor('cint1e_kin_sph')
    S1 = cell.pbc_intor('cint1e_ovlp_sph')
    N1 = mf.with_df.get_nuc()
    N2 = mf.with_df.get_pp()

    se = np.sum(mo.conj()*mo) * cell.vol/nbas
    print("overlap ", se, (moCoeff.T.conj().dot(S1.dot(moCoeff)))[0,0])

    kmo = Hv2(mo, JacDet, Jac, basMesh, G)/2.
    ke = np.sum(kmo.conj()*mo) * cell.vol/nbas
    print("kinetic ", ke, ke/se, (moCoeff.T.conj().dot(K1.dot(moCoeff)))[0,0])


    nuc =   ( Nuc1.dot( (Nuc1.T.dot(mo)*nucPot).conj() ) ).conj()/cell.vol
    #nuc = (Nuc1.conj().dot(Nuc2.T.dot(mo)))
    ne = np.sum(nuc.conj()*mo) * cell.vol/nbas
    print("nuclear ", ne, ne/se, (moCoeff.T.conj().dot(N2.dot(moCoeff)))[0,0])

    print(ke+ne, (ke+ne)/se, (moCoeff.T.conj().dot(K1.dot(moCoeff)))[0,0]+(moCoeff.T.conj().dot(N2.dot(moCoeff)))[0,0])
    #return
    
    from scipy.sparse.linalg import eigsh    
    
    #i = 0
    #def hv(v):
    #    return nucPot*v + Hv2(v, JacDet, Jac, basMesh, G)/2.
    #Ham   = LinearOperator((nbas, nbas), matvec = hv )        
    #(w,v) = eigsh(Ham, 1, which = 'SA')
    
    #print(w[0], cell.energy_nuc())
    
    nelec, nuclearPotential = cell.nelectron, cell.energy_nuc()
    
    def FC(C, J):
        C = np.asarray(C)
        Cout = 0.*np.asarray(C)

        if (len(C.shape) == 2 ):
            for i in range(C.shape[0]):
                Cout[i] = (J)*C[i] + ( Nuc1.dot( (Nuc1.T.dot(mo)*nucPot).conj() ) ).conj()/cell.vol + Hv2(C[i], JacDet, Jac, basMesh, G).real/2. 
            return Cout

        else :
            return (J)*C + ( Nuc1.dot( (Nuc1.T.dot(mo)*nucPot).conj() ) ).conj()/cell.vol + Hv2(C, JacDet, Jac, basMesh, G).real/2.         
    
    def precond(x, e0):
        return Condition(x, G2/2, basMesh, e0)
    
    Ck = np.random.random((nelec//2,nbas))
    #conv, e, x0 = Davidson.davidson1(lambda C : FC(C, 0.*G2), Ck, precond, nroots=nelec//2, verbose=0)
    #print(basMesh, conv, e)

    conv, e, orbs = Davidson.davidson1(lambda C : FC(C, 0.*G2), Ck, precond, nroots=nelec//2, verbose=0, tol=1e-3)
    orbs = np.asarray(orbs)
    
    density = 2*np.einsum('ig,ig->g', orbs, orbs.conj())

    davidsonTol = 1.e-3 
    j = V2e(density.real, 0.*G2, davidsonTol) * nbas/cell.vol
    Ho = FC(orbs, j/2.)
    Energy = 2.*np.einsum('ig,ig', orbs, Ho)  + nuclearPotential


    #dc = FDiisContext(10)

    from pyscf.lib.diis import DIIS    
    dc = DIIS()
    dc.space = 10
    

    t0 = time.time()
    for it in range(40):
        conv, e, orbs = Davidson.davidson1(lambda C : FC(C, j), orbs, precond, nroots=nelec//2, verbose=0, tol=davidsonTol)
        #orbs, e, iter, maxerror = Davidson.Davidson(lambda C : FC(C, j), precond, Ck)
        orbs = np.asarray(orbs)
        
        density = 2*np.einsum('ig,ig->g', orbs, orbs.conj())
        oldj = 1.*j
        j = V2e(density, j/(nbas/cell.vol)/4./np.pi) * nbas/cell.vol
        
        error = j - oldj
        
        oldE = Energy
        Ho = FC(orbs, j/2.)
        Energy = 2.*np.einsum('ig,ig', orbs, Ho)  + nuclearPotential
        errorNorm = np.linalg.norm(error)

        de = abs(Energy - oldE).real
        davidsonTol = max(1.e-10, min(davidsonTol, 0.001*de))

        dt = time.time() - t0
        print("{0:3d}   {1:15.8f}  {2:15.8f}  {3:15.8f}  {4:15.2f}".format(it, Energy.real, errorNorm, de, dt) )
        if (errorNorm < 1.e-5 or de < 1.e-8):
           break

        #j = dc.Apply(j, error)[0]
        j = dc.update(j, error)
    #'''
    return Energy.real , orbs 

