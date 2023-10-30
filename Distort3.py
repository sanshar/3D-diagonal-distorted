import os
import numpy as jnp
import matplotlib.pyplot as plt
from jax import lax, jit
from jax import numpy as jnp
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
from jax import grad, jit, vmap, jacfwd, jacrev
import numpy as np
import Distort, Distort2

# define Chebyshev polynomial evaluator
def gfun(x,a,b,nb):
  nx = x.shape[0]
  xc = jnp.clip(x,a,b)
  u = jnp.reshape( 2*(xc - (a+b)/2)/(b-a), (nx,1) )
  n = jnp.reshape( jnp.arange(nb), (1,nb) )
  g = jnp.cos(n*jnp.arccos(u))
  return g

def Gtildefun(x,a,b,nb):

  nx = x.shape[0]
  u = jnp.reshape( 2*(x - (a+b)/2)/(b-a), (nx,1) )
  G0 = u
  G1 = u**2/2

  g = gfun(x,a,b,nb+1)
  gplus = g[:,3:nb+1]
  gminus = g[:,1:nb-1]
  nvec = jnp.reshape(jnp.arange(2,nb),(1,nb-2))
  G2 = gplus/(2*(nvec+1)) - gminus/(2*(nvec-1))

  G = jnp.hstack((G0,G1,G2))
  return G

def Gfun(x,a,b,nb):
  Gtilde = Gtildefun(x,a,b,nb)
  Gtilde0 = Gtildefun(a*jnp.ones(1),a,b,nb)
  G = (b-a)*(Gtilde - Gtilde0)/2
  return G

# define Chebyshev polynomial derivative evaluator
def dgfun(x,a,b,nb):
  nx = x.shape[0]
  xc = jnp.clip(x,a+1e-10,b-1e-10)
  u = jnp.reshape( 2*(xc - (a+b)/2)/(b-a), (nx,1) )
  n = jnp.reshape( jnp.arange(nb), (1,nb) )
  #theta = jnp.arccos(u)
  dg = n*jnp.sin(n*jnp.arccos(u))/jnp.sin(jnp.arccos(u)) * 2 / (b-a)
  #dg = n*jnp.sin(n*jnp.arccos(u))/(1-u**2)**0.5
  return dg

# Input:
    # Cheb coefficients (C_T1,C_T2,C_T3) of coordinate functions
    # Product evaluation grid xp1 times xp2 times xp3
# Output:
    # T1fit,T2fit,T3fit, each of same shape as product grid
#@jit
def fastrak(C_T1,C_T2,C_T3,xp1,xp2,xp3,a1,b1,a2,b2,a3,b3):
    nb1,nb2,nb3 = C_T1.shape[0], C_T1.shape[1], C_T1.shape[2]

    gp1dx = gfun(np.asarray([a1,b1]), a1, b1, nb1)
    gp2dy = gfun(np.asarray([a2,b2]), a2, b2, nb2)
    gp3dz = gfun(np.asarray([a3,b3]), a3, b3, nb3)

    T1dx = jnp.einsum('abc,xa,yb,zc->xyz',C_T1,gp1dx,gp2dy,gp3dz)
    T2dx = jnp.einsum('abc,xa,yb,zc->xyz',C_T2,gp1dx,gp2dy,gp3dz)
    T3dx = jnp.einsum('abc,xa,yb,zc->xyz',C_T3,gp1dx,gp2dy,gp3dz)

    gp1 = gfun(xp1,a1,b1,nb1)
    gp2 = gfun(xp2,a2,b2,nb2)
    gp3 = gfun(xp3,a3,b3,nb3)

    T1fit = jnp.einsum('abc,xa,yb,zc->xyz',C_T1,gp1,gp2,gp3)
    T2fit = jnp.einsum('abc,xa,yb,zc->xyz',C_T2,gp1,gp2,gp3)
    T3fit = jnp.einsum('abc,xa,yb,zc->xyz',C_T3,gp1,gp2,gp3)

    # Pin the corners
    T1fit = (b1-a1) * ( T1fit - T1fit[0,0,0] )/(T1dx[-1,-1,-1] - T1dx[0,0,0]) + a1
    T2fit = (b2-a2) * ( T2fit - T2fit[0,0,0] )/(T2dx[-1,-1,-1] - T2dx[0,0,0]) + a2
    T3fit = (b3-a3) * ( T3fit - T3fit[0,0,0] )/(T3dx[-1,-1,-1] - T3dx[0,0,0]) + a3

    '''


    xp3 = xp3 + 1.e-4
    gp1 = gfun(xp1,a1,b1,nb1)
    gp2 = gfun(xp2,a2,b2,nb2)
    gp3 = gfun(xp3,a3,b3,nb3)

    T1fit22 = jnp.einsum('abc,xa,yb,zc->xyz',C_T1,gp1,gp2,gp3)
    T2fit22 = jnp.einsum('abc,xa,yb,zc->xyz',C_T2,gp1,gp2,gp3)
    T3fit22 = jnp.einsum('abc,xa,yb,zc->xyz',C_T3,gp1,gp2,gp3)

    # Pin the corners
    #T1fit22 = (b1-a1) * ( T1fit22 - T1fit[0,0,0] )/(T1dx[-1,-1,-1] - T1dx[0,0,0]) + a1
    #T2fit22 = (b2-a2) * ( T2fit22 - T2fit[0,0,0] )/(T2dx[-1,-1,-1] - T2dx[0,0,0]) + a2
    #T3fit22 = (b3-a3) * ( T3fit22 - T3fit[0,0,0] )/(T3dx[-1,-1,-1] - T3dx[0,0,0]) + a3

    print( "derivative ", (T1fit22.flatten()[1] - T1fit.flatten()[1])/1e-4 )
    print( "derivative ", (T2fit22.flatten()[1] - T2fit.flatten()[1])/1e-4 )
    print( "derivative ", (T3fit22.flatten()[1] - T3fit.flatten()[1])/1e-4 )
    


    
    xp3 = xp3 - 2.e-4
    gp1 = gfun(xp1,a1,b1,nb1)
    gp2 = gfun(xp2,a2,b2,nb2)
    gp3 = gfun(xp3,a3,b3,nb3)

    T1fit22b = jnp.einsum('abc,xa,yb,zc->xyz',C_T1,gp1,gp2,gp3)
    T2fit22b = jnp.einsum('abc,xa,yb,zc->xyz',C_T2,gp1,gp2,gp3)
    T3fit22b = jnp.einsum('abc,xa,yb,zc->xyz',C_T3,gp1,gp2,gp3)

    # Pin the corners
    #T1fit22b = (b1-a1) * ( T1fit22b - T1fit[0,0,0] )/(T1dx[-1,-1,-1] - T1dx[0,0,0]) + a1
    #T2fit22b = (b2-a2) * ( T2fit22b - T2fit[0,0,0] )/(T2dx[-1,-1,-1] - T2dx[0,0,0]) + a2
    #T3fit22b = (b3-a3) * ( T3fit22b - T3fit[0,0,0] )/(T3dx[-1,-1,-1] - T3dx[0,0,0]) + a3

    print(xp3)
    print(T3fit.flatten()[:5])
    print(T3fit22.flatten()[:5])
    print(T3fit22b.flatten()[:5])
    print( "derivative ", (T1fit22.flatten()[1] - T1fit22b.flatten()[1])/2./1e-4 )
    print( "derivative ", (T2fit22.flatten()[1] - T2fit22b.flatten()[1])/2./1e-4 )
    print( "derivative ", (T3fit22.flatten()[1] - T3fit22b.flatten()[1])/2./1e-4 )
    '''
    return T1fit,T2fit,T3fit


# Input:
    # Cheb coefficients (C_T1,C_T2,C_T3) of coordinate functions
    # Product evaluation grid xp1 times xp2 times xp3
# Output:
    # T1fit,T2fit,T3fit, each of same shape as product grid
#@jit
def fastrak_jac(C_T1,C_T2,C_T3,xp1,xp2,xp3,a1,b1,a2,b2,a3,b3):
    nb1,nb2,nb3 = C_T1.shape[0], C_T1.shape[1], C_T1.shape[2]

    gp1dx = gfun(np.asarray([a1,b1]), a1, b1, nb1)
    gp2dy = gfun(np.asarray([a2,b2]), a2, b2, nb2)
    gp3dz = gfun(np.asarray([a3,b3]), a3, b3, nb3)

    T1dx = jnp.einsum('abc,xa,yb,zc->xyz',C_T1,gp1dx,gp2dy,gp3dz)
    T2dx = jnp.einsum('abc,xa,yb,zc->xyz',C_T2,gp1dx,gp2dy,gp3dz)
    T3dx = jnp.einsum('abc,xa,yb,zc->xyz',C_T3,gp1dx,gp2dy,gp3dz)

    np1 = xp1.shape[0]
    np2 = xp2.shape[0]
    np3 = xp3.shape[0]

    gp1 = gfun(xp1,a1,b1,nb1)
    gp2 = gfun(xp2,a2,b2,nb2)
    gp3 = gfun(xp3,a3,b3,nb3)

    deltaT1 = (T1dx[-1,-1,-1] - T1dx[0,0,0])
    deltaT2 = (T2dx[-1,-1,-1] - T2dx[0,0,0])
    deltaT3 = (T3dx[-1,-1,-1] - T3dx[0,0,0])

    dgp1 = dgfun(xp1,a1,b1,nb1)
    dgp2 = dgfun(xp2,a2,b2,nb2)
    dgp3 = dgfun(xp3,a3,b3,nb3)

    J11fit = jnp.einsum('abc,xa,yb,zc->xyz',C_T1,dgp1,gp2,gp3)* (b1-a1) /deltaT1 
    J12fit = jnp.einsum('abc,xa,yb,zc->xyz',C_T1,gp1,dgp2,gp3)* (b1-a1) /deltaT1 
    J13fit = jnp.einsum('abc,xa,yb,zc->xyz',C_T1,gp1,gp2,dgp3)* (b1-a1) /deltaT1 

    J21fit = jnp.einsum('abc,xa,yb,zc->xyz',C_T2,dgp1,gp2,gp3)* (b2-a2) /deltaT2 
    J22fit = jnp.einsum('abc,xa,yb,zc->xyz',C_T2,gp1,dgp2,gp3)* (b2-a2) /deltaT2 
    J23fit = jnp.einsum('abc,xa,yb,zc->xyz',C_T2,gp1,gp2,dgp3)* (b2-a2) /deltaT2 

    J31fit = jnp.einsum('abc,xa,yb,zc->xyz',C_T3,dgp1,gp2,gp3)* (b3-a3) /deltaT3
    J32fit = jnp.einsum('abc,xa,yb,zc->xyz',C_T3,gp1,dgp2,gp3)* (b3-a3) /deltaT3
    J33fit = jnp.einsum('abc,xa,yb,zc->xyz',C_T3,gp1,gp2,dgp3)* (b3-a3) /deltaT3

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

# define Fourier mode evaluator
def ffun(x,a,b,nf):
    nx = x.shape[0]
    u = jnp.reshape( 2*jnp.pi*(x - (a+b)/2)/(b-a), (nx,1) )
    n = jnp.reshape( jnp.arange(-nf,nf+1), (1,2*nf+1) )
    f = jnp.exp(1j*n*u) / jnp.sqrt(b-a)
    return f

# define Fourier mode derivative evaluator (not properly normalized...)
def dffun(x,a,b,nf):
    nx = x.shape[0]
    u = jnp.reshape( 2*jnp.pi*(x - (a+b)/2)/(b-a), (nx,1) )
    n = jnp.reshape( jnp.arange(-nf,nf+1), (1,2*nf+1) )
    df = ((2*jnp.pi)/(b-a)) * 1j*n * jnp.exp(1j*n*u) / jnp.sqrt(b-a)
    return df

@jit
def fourier_fastrak(C_T1,C_T2,C_T3,xp1,xp2,xp3,a1,b1,a2,b2,a3,b3):

    np1 = xp1.shape[0]
    np2 = xp2.shape[0]
    np3 = xp3.shape[0]

    nf1 = (C_T1.shape[0] - 1)//2
    nf2 = (C_T1.shape[1] - 1)//2
    nf3 = (C_T1.shape[2] - 1)//2

    fp1 = ffun(xp1,a1,b1,nf1)
    fp2 = ffun(xp2,a2,b2,nf2)
    fp3 = ffun(xp3,a3,b3,nf3)

    T1fit = jnp.reshape(xp1,(np1,1,1)) + jnp.real ( jnp.einsum('abc,xa,yb,zc->xyz',C_T1,fp1,fp2,fp3) )
    T2fit = jnp.reshape(xp2,(1,np2,1)) + jnp.real ( jnp.einsum('abc,xa,yb,zc->xyz',C_T2,fp1,fp2,fp3) )
    T3fit = jnp.reshape(xp3,(1,1,np3)) + jnp.real ( jnp.einsum('abc,xa,yb,zc->xyz',C_T3,fp1,fp2,fp3) )

    return T1fit,T2fit,T3fit


# Input:
    # Fourier coefficients (C_T1,C_T2,C_T3) of coordinate functions
    # Product evaluation grid xp1 times xp2 times xp3
# Output:
    # T1fit,T2fit,T3fit, each of same shape as product grid
#@jit
def fourier_fastrak_jac(C_T1,C_T2,C_T3,xp1,xp2,xp3,a1,b1,a2,b2,a3,b3):

    np1 = xp1.shape[0]
    np2 = xp2.shape[0]
    np3 = xp3.shape[0]

    nf1 = (C_T1.shape[0] - 1)//2
    nf2 = (C_T1.shape[1] - 1)//2
    nf3 = (C_T1.shape[2] - 1)//2

    fp1 = ffun(xp1,a1,b1,nf1)
    fp2 = ffun(xp2,a2,b2,nf2)
    fp3 = ffun(xp3,a3,b3,nf3)

    dfp1 = dffun(xp1,a1,b1,nf1)
    dfp2 = dffun(xp2,a2,b2,nf2)
    dfp3 = dffun(xp3,a3,b3,nf3)


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

def returnFunc2(cell, Cx, Cy, Cz):
    A = cell.lattice_vectors()
    a1,b1,a2,b2,a3,b3 = 0.,A[0,0],0.,A[1,1],0.,A[2,2]


    def invFlowSinglePoint(gx, gy, gz):
        Tbasx, Tbasy, Tbasz = fastrak(Cx, Cy, Cz, np.asarray([gx]), np.asarray([gy]), np.asarray([gz]), a1, b1, a2, b2, a3, b3)
        #Tbasx, Tbasy, Tbasz = Tbasx[:nmesh[0],:nmesh[1],:nmesh[2]], Tbasy[:nmesh[0],:nmesh[1],:nmesh[2]], Tbasz[:nmesh[0],:nmesh[1],:nmesh[2]]
        return Tbasx[0], Tbasy[0], Tbasz[0]

    def invFlow(nmesh):
        gx = np.linspace(a1,b1,nmesh[0]+1)#+1.e-6
        gy = np.linspace(a2,b2,nmesh[1]+1)#+1.e-6
        gz = np.linspace(a3,b3,nmesh[2]+1)#+1.e-6
        Tbasx, Tbasy, Tbasz = fastrak(Cx, Cy, Cz, gx, gy, gz, a1, b1, a2, b2, a3, b3)
        Tbasx, Tbasy, Tbasz = Tbasx[:nmesh[0],:nmesh[1],:nmesh[2]], Tbasy[:nmesh[0],:nmesh[1],:nmesh[2]], Tbasz[:nmesh[0],:nmesh[1],:nmesh[2]]
        return np.hstack((Tbasx.reshape(-1,1), Tbasy.reshape(-1,1), Tbasz.reshape(-1,1)))

        
    #Cinvx = np.load("C_T1.npy")
    #Cinvy = np.load("C_T2.npy")
    #Cinvz = np.load("C_T3.npy")

    def JacAllFun(nmesh):
        gx = np.linspace(a1,b1,nmesh[0]+1)#+1.e-6
        gy = np.linspace(a2,b2,nmesh[1]+1)#+1.e-6
        gz = np.linspace(a3,b3,nmesh[2]+1)#+1.e-6
        DT, J = fastrak_jac(Cx, Cy, Cz, gx, gy, gz, a1, b1, a2, b2, a3, b3)
        DT = DT[:nmesh[0],:nmesh[1],:nmesh[2]]
        DT = DT.reshape(-1,3,3)
        Jac = np.asarray([np.linalg.inv(Ji) for Ji in DT])

        #JacDet = np.asarray([np.linalg.det(Ji) for Ji in Jac])
        #factor = np.sum(1./JacDet)/np.prod(nmesh)
        #Jac = Jac * factor

        return Jac

    def JacAllSinglePoint(gx, gy, gz):
        DT, J = fastrak_jac(Cx, Cy, Cz, np.asarray([gx]).reshape(-1,), np.asarray([gy]).reshape(-1,), np.asarray([gz]).reshape(-1,), a1, b1, a2, b2, a3, b3)
        DT = DT[:1,:1,:1]
        DT = DT.reshape(-1,3,3)
        Jac = np.asarray([np.linalg.inv(Ji) for Ji in DT])

        #JacDet = np.asarray([np.linalg.det(Ji) for Ji in Jac])
        #factor = np.sum(1./JacDet)/np.prod(nmesh)
        #Jac = Jac * factor

        return Jac

            
    return invFlow, JacAllFun, invFlowSinglePoint, JacAllSinglePoint


def returnFuncFourier(cell, Cx, Cy, Cz):
    A = cell.lattice_vectors()
    a1,b1,a2,b2,a3,b3 = 0.,A[0,0],0.,A[1,1],0.,A[2,2]


    def invFlowSinglePoint(gx, gy, gz):
        Tbasx, Tbasy, Tbasz = fourier_fastrak(Cx, Cy, Cz, np.asarray([gx]).reshape(-1,), np.asarray([gy]).reshape(-1,), np.asarray([gz]).reshape(-1,), a1, b1, a2, b2, a3, b3)
        return float(Tbasx), float(Tbasy), float(Tbasz)

    def invFlow(nmesh):
        gx = np.linspace(a1,b1,nmesh[0]+1)#+1.e-6
        gy = np.linspace(a2,b2,nmesh[1]+1)#+1.e-6
        gz = np.linspace(a3,b3,nmesh[2]+1)#+1.e-6
        Tbasx, Tbasy, Tbasz = fourier_fastrak(Cx, Cy, Cz, gx, gy, gz, a1, b1, a2, b2, a3, b3)
        Tbasx, Tbasy, Tbasz = Tbasx[:nmesh[0],:nmesh[1],:nmesh[2]], Tbasy[:nmesh[0],:nmesh[1],:nmesh[2]], Tbasz[:nmesh[0],:nmesh[1],:nmesh[2]]
        return np.hstack((Tbasx.reshape(-1,1), Tbasy.reshape(-1,1), Tbasz.reshape(-1,1)))

    #Cinvx = np.load("C_T1.npy")
    #Cinvy = np.load("C_T2.npy")
    #Cinvz = np.load("C_T3.npy")

    def JacAllFun(nmesh):
        gx = np.linspace(a1,b1,nmesh[0]+1)#+1.e-6
        gy = np.linspace(a2,b2,nmesh[1]+1)#+1.e-6
        gz = np.linspace(a3,b3,nmesh[2]+1)#+1.e-6
        DT, J = fourier_fastrak_jac(Cx, Cy, Cz, gx, gy, gz, a1, b1, a2, b2, a3, b3)
        DT = DT[:nmesh[0],:nmesh[1],:nmesh[2]]
        DT = DT.reshape(-1,3,3)
        Jac = np.asarray([np.linalg.inv(Ji) for Ji in DT])

        #JacDet = np.asarray([np.linalg.det(Ji) for Ji in Jac])
        #factor = np.sum(1./JacDet)/np.prod(nmesh)
        #Jac = Jac * factor

        return Jac

    def JacAllSinglePoint(gx, gy, gz):
        DT, J = fourier_fastrak_jac(Cx, Cy, Cz, np.asarray([gx]).reshape(-1,), np.asarray([gy]).reshape(-1,), np.asarray([gz]).reshape(-1,), a1, b1, a2, b2, a3, b3)
        DT = DT[:1,:1,:1]
        DT = DT.reshape(3,3)
        Jac = np.linalg.inv(DT)

        return Jac


            
    return invFlow, JacAllFun, invFlowSinglePoint, JacAllSinglePoint

    
def returnFunc(cell):
    A = cell.lattice_vectors()
    a1,b1,a2,b2,a3,b3 = 0.,A[0,0],0.,A[1,1],0.,A[2,2]
    C = np.load("cheb.npy")
    #C = np.fromfile("cheb.bin", dtype=np.float64).reshape(6, 6, 6, -1)


    if (C.shape[3]%3 != 0):
        print("the number of layers should be multiple of 3")
        exit(0)


    maxIter = 200
    alpha = 0.8
    tol = 1e-8

    def invFlow(grid):
        Tbasx, Tbasy, Tbasz, Jinv2 = Distort.inv_flow(grid[:,0], grid[:,1], grid[:,2], C, a1, b1, a2, b2, a3, b3, maxIter, alpha, tol)
        return np.hstack((Tbasx.reshape(-1,1), Tbasy.reshape(-1,1), Tbasz.reshape(-1,1)))

    Tdel = jacfwd(lambda grid : Distort2.flow(grid,C, a1,b1,a2,b2,a3,b3))
    JacFun = vmap(Tdel, in_axes=(0))
    
    def JacAllFun(grid):
        return JacFun(grid+1e-6)
    
    return invFlow, JacAllFun
    
    
def create_cond_and_body1(y1,y2,y3,invFlow,maxIter,alpha,tol):

  @jit
  def body_fun(carry):
    x1,x2,x3,T1,T2,T3,rho,iter = carry
    x1 = x1 - alpha*(T1-y1)
    x2 = x2 - alpha*(T2-y2)
    x3 = x3 - alpha*(T3-y3)
    T1,T2,T3,rho = knothe3dcheb(x1,x2,x3,a1,b1,a2,b2,a3,b3,c)
    return (x1,x2,x3,T1,T2,T3,rho,iter+1)

  @jit
  def cond_fun(carry):
    x1,x2,x3,T1,T2,T3,rho,iter = carry
    relErr1 = jnp.max(jnp.abs(T1-y1))/(b1-a1)
    relErr2 = jnp.max(jnp.abs(T2-y2))/(b2-a2)
    relErr3 = jnp.max(jnp.abs(T3-y3))/(b3-a3)
    cond1 = relErr1 > tol
    cond2 = relErr2 > tol
    cond3 = relErr3 > tol
    cond4 = iter < maxIter
    return (cond1+cond2+cond3)*cond4

  return body_fun, cond_fun
