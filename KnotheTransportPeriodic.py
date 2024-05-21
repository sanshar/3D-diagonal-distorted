#@title Import dependencies

import os
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=6'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
#os.environ['JAX_DISABLE_JIT'] = 'True'
from jax import numpy as jnp
from jax import scipy as jsp
from jax import value_and_grad, vmap, lax, jit, random, jacfwd, jacrev
#from jax.experimental.sparse import BCOO
#from jax.experimental.sparse import BCSR

import jax, jaxopt
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
#import cupy as cp

#from cupyx.scipy.sparse import coo_matrix, csr_matrix

from jax.config import config
config.update("jax_enable_x64", True)

import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds." %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def rhofun_fake(x1, x2, x3, mf, shift):
  return rhofun(x1, x2, x3, mf, shift)
 
# specify density
#jit
def rhofun(x1,x2,x3, mf, shift):

  A = mf.cell.lattice_vectors()
  a1,b1,a2,b2,a3,b3 = 0.,A[0,0],0.,A[1,1],0.,A[2,2]

  atoms = mf.mol._atm
  Zi = atoms[:,0]
  
  density = 0
  #'''
  for a in range(Zi.shape[0]):
    pos = mf.mol._env[atoms[a,1]:atoms[a,1]+3]

    #xx1, xx2, xx3 = Zi[a]*(x1-pos[0]), Zi[a]*(x2-pos[1]), Zi[a]*(x3-pos[2])
    #density += 1/(xx1**2 + xx2**2 + xx3**2 + 0.1**2)**0.5

    for n1 in range(-1,2):
      for n2 in range(-1,2):
        for n3 in range(-1,2):
          xx1, xx2, xx3 = (x1-pos[0]+n1*b1), (x2-pos[1]+n2*b2), (x3-pos[2]+n3*b3)
          #density += 1/(xx1**2 + xx2**2 + xx3**2 + (0.1/Zi[a])**2)**0.5

          #density += 1./((xx1**2 + (0.5/Zi[a])**2)/(xx2**2 + (0.5/Zi[a])**2)/(xx3**2 + (0.5/Zi[a])**2))**0.5
          #density += 1./((xx1**2 + (0.1/Zi[a])**2)*(xx2**2 + (0.1/Zi[a])**2)*(xx3**2 + (0.1/Zi[a])**2))**0.5

          r = (xx1**2 + xx2**2 + xx3**2)**0.5
          density += (jsp.special.erf((r+1e-6) * Zi[a] / 0.1) - jsp.special.erf((r+1e-6) /2.))/(r+1e-6)
          #density += (jsp.special.erf((xx1+1e-6) * Zi[a] / 0.5) - jsp.special.erf((xx1+1e-6) /1.))/(xx1+1e-6) + 0.05**(1./3) \
          #  * (jsp.special.erf((xx2+1e-6) * Zi[a] / 0.5) - jsp.special.erf((xx2+1e-6) /1.))/(xx2+1e-6) + 0.05**(1./3) \
          #  * (jsp.special.erf((xx3+1e-6) * Zi[a] / 0.5) - jsp.special.erf((xx3+1e-6) /1.))/(xx3+1e-6) + 0.05**(1./3) 

          #density = 1.
  return density + 0.03
  #'''
  #return 1./mf.cell.vol
  #xx1, xx2, xx3 = x1-3, x2-3, x3-3.
  #return 1./(xx1**2 + xx2**2 + xx3**2 + 0.1**2)**0.5
  
  #xx1, xx2, xx3 = x1-6, x2-4, x3-4.
  #rho  = 1./(xx1**2 + xx2**2 + xx3**2 + 0.1**2)**0.5/3.

  #xx1, xx2, xx3 = x1-2., x2-4, x3-4.
  #rho += 1./(xx1**2 + xx2**2 + xx3**2 + 0.1**2)**0.5/3.

  #xx1, xx2, xx3 = x1-4., x2-6, x3-4.
  #rho += 1./(xx1**2 + xx2**2 + xx3**2 + 0.1**2)**0.5/3.

  #rho = jnp.exp(-3.*(0.5**2 + (xx1)**2 + (xx2)**2 + (xx3)**2)**0.5)/jnp.exp(-1.5)/(0.5**2 + (xx1)**2 + (xx2)**2 + (xx3)**2 )**0.5 + 1.0
  #rho = jnp.exp(-0.9 * ((xx1)**2 + (xx2)**2 + (xx3)**2))/(0.05**2 + (xx1)**2 + (xx2)**2 + (xx3)**2 )**0.5  + 1.e-1 #+ 1.0
  #return rho

  nelec = mf.cell.nelectron

  if isinstance(mf.mo_coeff, list):
    moCoeff = mf.mo_coeff[0][:,:nelec//2].real
  else:
    moCoeff = mf.mo_coeff[:,:nelec//2].real
    
  coords = np.hstack((x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1)))
  ao = mf.cell.pbc_eval_gto('GTOval', coords)

  
  mo = np.einsum('ab,ra->br', moCoeff, ao)
  moe = mf.mo_energy[:nelec//2].real
  moe = abs(moe/moe[0])
  density = np.einsum('a, ar,ar->r', abs(moe), mo, mo)   + 0.01

  #density = np.ones((mo.shape[1],))
  return density
  
  #rho = ( jnp.exp(-3*jnp.sqrt(.3 + x1**2+(x2-1.0)**2 + (x3+1.0)**2)) + .01)
  #rho = ( jnp.exp(-3*jnp.sqrt(1 + x1**2+(x2-1.0)**2+(x3-0.0)**2))
  #+ jnp.exp(-3*jnp.sqrt(1 + (x1-1.0)**2+(x2+1.0)**2+(x3-0.0)**2))
  #+ jnp.exp(-3*jnp.sqrt(1 + (x1+1.0)**2+(x2+1.0)**2+(x3-0.0)**2)) + .01 )
  xx1, xx2, xx3 = x1-5., x2-5., x3-5.
    #rho = jnp.exp(-3.*(0.5**2 + (xx1)**2 + (xx2)**2 + (xx3)**2)**0.5)/jnp.exp(-1.5)/(0.5**2 + (xx1)**2 + (xx2)**2 + (xx3)**2 )**0.5 + 1.0
  rho = jnp.exp(-0.5 * ((xx1)**2 + (xx2)**2 + (xx3)**2))/(0.3**2 + (xx1)**2 + (xx2)**2 + (xx3)**2 )**0.5  +1.e-2#+ 1.0

  return rho


# number of Gauss quadrature points per dimension
#ng1 = 10  # dimension 1
#ng2 = 10  # dimension 2
#ng3 = 10  # dimension 3

# number of fitting Chebyshev polynomials per dimension
#nb1 = 10  # dimension 1
#nb2 = 10  # dimension 2
#nb3 = 10  # dimension 3

# hyperparameters for fixed-point iteration computation of inverse flow
maxIter = 1000 # maximum number of iterations
alpha = 1. # mixing parameter
tol = 1e-10 # convergence tolerance




#@title Set up and define functions

# define Nth root of target density
#jit
def rhorootfun(x1,x2,x3,mf,N, shift):
  return jnp.exp(jnp.log(rhofun(x1,x2,x3, mf, shift))/N)

'''
'''

# define Chebyshev polynomial evaluator
def gfun(x,a,b,nb):
  nx = x.shape[0]
  u = jnp.reshape( (x-a)/(b-a), (nx,1) )
  n = jnp.reshape( jnp.arange(nb), (1,nb) ) - nb//2
  g = jnp.exp(-2*jnp.pi*1j*n*u)
  return g

'''
'''

def getJac(Xp1, Xp2, Xp3, C,a1,b1,a2,b2,a3,b3,N):
  def fun(X):
    T1, T2, T3, _ = flowJittable(X[:1], X[1:2], X[2:], C,a1,b1,a2,b2,a3,b3,N)
    return jnp.asarray([T1[0], T2[0], T3[0]])
  
  J = jax.lax.map(jacfwd(fun), jnp.hstack((Xp1.reshape(-1,1), Xp2.reshape(-1,1), Xp3.reshape(-1,1))) )
  return J
    
# define Chebyshev antiderivative evaluator
def Gfun(x,a,b,nb):

  nx = x.shape[0]
  u = jnp.reshape( (x-a)/(b-a), (nx,1) )

  nL = jnp.arange(0,nb//2) - nb//2
  nR = jnp.arange(nb//2 + 1, nb ) - nb//2

  GL = (jnp.exp(-2*jnp.pi*1j*nL*u) - 1)/(-2*jnp.pi*1j*nL)
  G0 = u
  GR = (jnp.exp(-2*jnp.pi*1j*nR*u) - 1)/(-2*jnp.pi*1j*nR)

  G = (b-a) * jnp.hstack((GL,G0,GR))
  return G

# define Chebyshev fit function
@jit
def chebfit(F,g1,g2,g3,S):
  c = jnp.einsum('ijk,ia,jb,kc->abc',F,g1,g2,g3) / S
  c = jnp.conj(c)
  return c

# define Knothe transport function
@jit
def knothe3dcheb(x1,x2,x3,a1,b1,a2,b2,a3,b3,c):

  L1 = b1-a1
  L2 = b2-a2
  L3 = b3-a3
  nb1 = c.shape[0]
  nb2 = c.shape[1]
  nb3 = c.shape[2]
  m1 = jnp.ravel( Gfun(b1*jnp.ones(1),a1,b1,nb1) )
  m2 = jnp.ravel( Gfun(b2*jnp.ones(1),a2,b2,nb2) )
  m3 = jnp.ravel( Gfun(b3*jnp.ones(1),a3,b3,nb3) )

  M = jnp.einsum('abc,a,b,c',c,m1,m2,m3)
  cc = c/M

  g1 = gfun(x1,a1,b1,nb1)
  g2 = gfun(x2,a2,b2,nb2)
  g3 = gfun(x3,a3,b3,nb3)
  G1 = Gfun(x1,a1,b1,nb1)
  G2 = Gfun(x2,a2,b2,nb2)
  G3 = Gfun(x3,a3,b3,nb3)

  rho1 = jnp.einsum('abc,ia,b,c->i',cc,g1,m2,m3)
  rho2 = jnp.einsum('abc,ia,ib,c->i',cc,g1,g2,m3)
  rho = jnp.einsum('abc,ia,ib,ic->i',cc,g1,g2,g3)

  T1 = a1 + L1*jnp.einsum('abc,ia,b,c->i',cc,G1,m2,m3)
  T2 = a2 + L2*jnp.einsum('abc,ia,ib,c->i',cc,g1,G2,m3)/rho1
  T3 = a3 + L3*jnp.einsum('abc,ia,ib,ic->i',cc,g1,g2,G3)/rho2

  T1 = jnp.real(T1)
  T2 = jnp.real(T2)
  T3 = jnp.real(T3)
  rho = jnp.real(rho)
  
  return T1,T2,T3,rho

@jit
def errFun(allx, allT, c,a1,b1,a2,b2,a3,b3):
    N = allx.shape[0]//3
    x1 = allx[:N]
    x2 = allx[N:2*N]
    x3 = allx[2*N:]

    T1, T2, T3, rho = knothe3dcheb(x1, x2, x3, a1, b1, a2, b2, a3, b3,c)
    return jnp.linalg.norm((allT - jnp.hstack((T1, T2, T3)) ))



# define inverse Knothe transport function
@jit
def inv_knothe3dcheb(y1,y2,y3,a1,b1,a2,b2,a3,b3,c,maxIter,alpha,tol):
  body_fun, cond_fun = create_cond_and_body(y1,y2,y3,a1,b1,a2,b2,a3,b3,c,maxIter,alpha,tol)
  x1 = y1
  x2 = y2
  x3 = y3
  T1, T2, T3, rho = knothe3dcheb(x1,x2,x3,a1,b1,a2,b2,a3,b3,c)


  '''
  solver = jaxopt.GradientDescent(fun=errFun, maxiter=maxIter)
  #solver = jaxopt.NonlinearCG(fun=errFun, method="polak-ribiere", maxiter=maxIter)
  #solver = jaxopt.LBFGS(fun=errFun, maxiter=maxIter)
  res = solver.run(jnp.hstack((T1,T2,T3)), jnp.hstack((y1,y2,y3)), c,a1,b1,a2,b2,a3,b3)      
  N = y1.shape[0]
  allS = res.params
  return allS[:N], allS[N:2*N], allS[2*N:], 1./rho, 1
  '''

  init_val = (x1,x2,x3,T1,T2,T3,rho,0)
  x1,x2,x3,T1,T2,T3,rho,iter = lax.while_loop(cond_fun, body_fun, init_val)
  return x1,x2,x3,1/rho,iter

def create_cond_and_body(y1,y2,y3,a1,b1,a2,b2,a3,b3,c,maxIter,alpha,tol):

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

@jit
def flowJittable(x1, x2, x3, C, a1, b1, a2, b2,a3,b3,N):
  T1 = x1
  T2 = x2
  T3 = x3
  p = x1.shape[0]
  J = jnp.ones(p)/p

  N1, N2, N3 = C.shape[0], C.shape[1], C.shape[2]
  cloop = 1.*C.reshape(N1, N2, N3, -1, 3).transpose([3,4,0,1,2])

  def fun(carry, cn):
    T1, T2, T3, J = carry[0], carry[1], carry[2], carry[3]

    T1,T2,T3,rhofac = knothe3dcheb(T1,T2,T3,a1,b1,a2,b2,a3,b3,cn[0])
    J = J * rhofac

    T3,T1,T2,rhofac = knothe3dcheb(T3,T1,T2,a3,b3,a1,b1,a2,b2,jnp.transpose(cn[1],(2,0,1)))
    J = J * rhofac

    T2,T3,T1,rhofac = knothe3dcheb(T2,T3,T1,a2,b2,a3,b3,a1,b1,jnp.transpose(cn[2],(1,2,0)))
    J = J * rhofac

    J = J / jnp.sum(J)
    return (T1, T2, T3, J), 0.

  carry, _ = jax.lax.scan(fun, (T1,T2,T3,J),cloop)
  return carry[0], carry[1], carry[2], carry[3]

# Define function that evaluates flow map and Jacobian of input
#@jit
def flow(x1,x2,x3,C,a1,b1,a2,b2,a3,b3,N):
  T1 = x1
  T2 = x2
  T3 = x3
  p = x1.shape[0]
  J = jnp.ones(p)/p
  for n in range(N):
    c = C[:,:,:,n]
    if jnp.mod(n,3)==0:
      T1,T2,T3,rhofac = knothe3dcheb(T1,T2,T3,a1,b1,a2,b2,a3,b3,c)
    elif jnp.mod(n,3)==1:
      T3,T1,T2,rhofac = knothe3dcheb(T3,T1,T2,a3,b3,a1,b1,a2,b2,jnp.transpose(c,(2,0,1)))
    else:
      T2,T3,T1,rhofac = knothe3dcheb(T2,T3,T1,a2,b2,a3,b3,a1,b1,jnp.transpose(c,(1,2,0)))
    J = J * rhofac
    J = J / jnp.sum(J)
  return T1,T2,T3,J

# Define function that evaluates flow map and Jacobian of input
#@jit
def inv_flow(x1,x2,x3,C,a1,b1,a2,b2,a3,b3,N, maxIter=maxIter,alpha=alpha,tol=tol):
  T1 = x1
  T2 = x2
  T3 = x3
  p = x1.shape[0]
  Jinv = jnp.ones(p)/p
  for n in range(N):
    c = C[:,:,:,N-n-1]
    if jnp.mod(N-n-1,3)==0:
      T1,T2,T3,rhoinvfac,_ = inv_knothe3dcheb(T1,T2,T3,a1,b1,a2,b2,a3,b3,c,maxIter,alpha,tol)
    elif jnp.mod(N-n-1,3)==1:
      T3,T1,T2,rhoinvfac,_ = inv_knothe3dcheb(T3,T1,T2,a3,b3,a1,b1,a2,b2,jnp.transpose(c,(2,0,1)),maxIter,alpha,tol)
    else:
      T2,T3,T1,rhoinvfac,_ = inv_knothe3dcheb(T2,T3,T1,a2,b2,a3,b3,a1,b1,jnp.transpose(c,(1,2,0)),maxIter,alpha,tol)
    Jinv = Jinv * rhoinvfac
    Jinv = Jinv / jnp.sum(Jinv)
  return T1,T2,T3,Jinv


def LearnTransportInverse(nb1, nb2, nb3, mf, N, shift):
    a1,b1,a2,b2,a3,b3 = 0.,mf.cell.a[0,0],0.,mf.cell.a[1,1],0.,mf.cell.a[2,2]

    C = jnp.zeros((nb1,nb2,nb3,N),dtype=jnp.complex128)

    ng1, ng2, ng3 = nb1, nb2, nb3
    #ng1, ng2, ng3 = 3*nb1//2, 3*nb2//2, 3*nb3//2


    # define Gauss quadrature grid
    u1 = jnp.linspace(0,ng1-1,ng1)/ng1
    xg1 = a1 + (b1-a1)*u1
    u2 = jnp.linspace(0,ng2-1,ng2)/ng2
    xg2 = a2 + (b2-a2)*u2
    u3 = jnp.linspace(0,ng3-1,ng3)/ng3
    xg3 = a3 + (b3-a3)*u3

    X1 = jnp.repeat(jnp.repeat(jnp.reshape(xg1,(ng1,1,1)),ng2,axis=1),ng3,axis=2)
    X2 = jnp.repeat(jnp.repeat(jnp.reshape(xg2,(1,ng2,1)),ng1,axis=0),ng3,axis=2)
    X3 = jnp.repeat(jnp.repeat(jnp.reshape(xg3,(1,1,ng3)),ng1,axis=0),ng2,axis=1)

    Xg1 = jnp.reshape(X1,ng1*ng2*ng3)
    Xg2 = jnp.reshape(X2,ng1*ng2*ng3)
    Xg3 = jnp.reshape(X3,ng1*ng2*ng3)

    gX1 = gfun(Xg1,a1,b1,nb1)
    gX2 = gfun(Xg2,a2,b2,nb2)
    gX3 = gfun(Xg3,a3,b3,nb3)


    Tg1, Tg2, Tg3 = Xg1, Xg2, Xg3

  # evaluate Chebyshev polynomials on quadrature grid
    g1 = gfun(xg1,a1,b1,nb1)
    g2 = gfun(xg2,a2,b2,nb2)
    g3 = gfun(xg3,a3,b3,nb3)

    S = 1

    for megaIter in range(25):
      Tg1prev, Tg2prev, Tg3prev = Tg1, Tg2, Tg3

      itervec = jnp.zeros(N-1)

      f = 1./rhorootfun(Tg1,Tg2,Tg3,mf,N,0.)
      F = jnp.reshape(f,(ng1,ng2,ng3))

      print("Maximum relative fitting error per flow step:")
      for n in range(N):
        c = chebfit(F,g1,g2,g3,S)

        ffit = jnp.einsum('abc,ia,ib,ic->i',c,gX1,gX2,gX3)
        Ffit = jnp.reshape(ffit,(ng1,ng2,ng3))
        Fhat = F/jnp.sum(F)
        Ffithat = Ffit/jnp.sum(Ffit)
        print(jnp.max(jnp.abs(Ffithat-Fhat))/jnp.max(Fhat))

        c = c/jnp.linalg.norm(c)
        C = C.at[:,:,:,n].set(c)

        if n<N-1:
          if jnp.mod(n,3)==0:
            Z1,Z2,Z3,_,iter = inv_knothe3dcheb(Xg1,Xg2,Xg3,a1,b1,a2,b2,a3,b3,c,maxIter,alpha,tol)
          elif jnp.mod(n,3)==1:
            Z3,Z1,Z2,_,iter = inv_knothe3dcheb(Xg3,Xg1,Xg2,a3,b3,a1,b1,a2,b2,jnp.transpose(c,(2,0,1)),maxIter,alpha,tol)
          else:
            Z2,Z3,Z1,_,iter = inv_knothe3dcheb(Xg2,Xg3,Xg1,a2,b2,a3,b3,a1,b1,jnp.transpose(c,(1,2,0)),maxIter,alpha,tol)
          itervec = itervec.at[n].set(iter)
          gZ1 = gfun(Z1,a1,b1,nb1)
          gZ2 = gfun(Z2,a2,b2,nb2)
          gZ3 = gfun(Z3,a3,b3,nb3)
          f = jnp.einsum('abc,ia,ib,ic->i',c,gZ1,gZ2,gZ3)
          F = jnp.reshape(f,(ng1,ng2,ng3))

      Tg1,Tg2,Tg3,_ = flow(Xg1,Xg2,Xg3,C,a1,b1,a2,b2,a3,b3,N)

      print("")
      print("Number of fixed-point iterations for inversion at each flow step:")
      print(itervec.astype(int))

      T1err = jnp.linalg.norm(Tg1-Tg1prev)/jnp.linalg.norm(Tg1prev)
      T2err = jnp.linalg.norm(Tg2-Tg2prev)/jnp.linalg.norm(Tg2prev)
      T3err = jnp.linalg.norm(Tg3-Tg3prev)/jnp.linalg.norm(Tg3prev)
      print(T1err, T2err, T3err)
      if (max(T1err, T2err, T3err) < tol):
        break
    
    return C



#@title Learn the transport
def LearnTransport(nb1, nb2, nb3, mf, N, shift):
    a1,b1,a2,b2,a3,b3 = 0.,mf.cell.a[0,0],0.,mf.cell.a[1,1],0.,mf.cell.a[2,2]

    #ng1, ng2, ng3 = 2*nb1, 2*nb2, 2*nb3
    ng1, ng2, ng3 = nb1, nb2, nb3

    # define Gauss quadrature grid
    u1 = jnp.linspace(0,ng1-1,ng1)/ng1
    xg1 = a1 + (b1-a1)*u1
    u2 = jnp.linspace(0,ng2-1,ng2)/ng2
    xg2 = a2 + (b2-a2)*u2
    u3 = jnp.linspace(0,ng3-1,ng3)/ng3
    xg3 = a3 + (b3-a3)*u3

    X1 = jnp.repeat(jnp.repeat(jnp.reshape(xg1,(ng1,1,1)),ng2,axis=1),ng3,axis=2)
    X2 = jnp.repeat(jnp.repeat(jnp.reshape(xg2,(1,ng2,1)),ng1,axis=0),ng3,axis=2)
    X3 = jnp.repeat(jnp.repeat(jnp.reshape(xg3,(1,1,ng3)),ng1,axis=0),ng2,axis=1)

    Xg1 = jnp.reshape(X1,ng1*ng2*ng3)
    Xg2 = jnp.reshape(X2,ng1*ng2*ng3)
    Xg3 = jnp.reshape(X3,ng1*ng2*ng3)

  # evaluate Chebyshev polynomials on quadrature grid
    g1 = gfun(xg1,a1,b1,nb1)
    g2 = gfun(xg2,a2,b2,nb2)
    g3 = gfun(xg3,a3,b3,nb3)

    S = 1


    # Coefficients encode learned transport
    C = jnp.zeros((nb1,nb2,nb3,N),dtype=jnp.complex128)
    itervec = jnp.zeros(N-1)
    f = rhorootfun(Xg1,Xg2,Xg3,mf,N, shift)
    F = jnp.reshape(f,(ng1,ng2,ng3))
    gX1 = gfun(Xg1,a1,b1,nb1)
    gX2 = gfun(Xg2,a2,b2,nb2)
    gX3 = gfun(Xg3,a3,b3,nb3)

    print("Maximum relative and absolute fitting error per flow step:")
    for n in range(N):
        c = chebfit(F,g1,g2,g3,S)
        ffit = jnp.einsum('abc,ia,ib,ic->i',c,gX1,gX2,gX3)
        Ffit = jnp.reshape(ffit,(ng1,ng2,ng3))
        Fhat = F/jnp.sum(F)
        Ffithat = Ffit/jnp.sum(Ffit)
        print(jnp.max(jnp.abs(Ffithat-Fhat))/jnp.max(Fhat), jnp.max(jnp.abs(Ffithat-Fhat)))

        c = c/jnp.linalg.norm(c)
        C = C.at[:,:,:,n].set(c)

        if n<N-1:
            if jnp.mod(n,3)==0:
                Z1,Z2,Z3,_,iter = inv_knothe3dcheb(Xg1,Xg2,Xg3,a1,b1,a2,b2,a3,b3,c,maxIter,alpha,tol)
            elif jnp.mod(n,3)==1:
                Z3,Z1,Z2,_,iter = inv_knothe3dcheb(Xg3,Xg1,Xg2,a3,b3,a1,b1,a2,b2,jnp.transpose(c,(2,0,1)),maxIter,alpha,tol)
            else:
                Z2,Z3,Z1,_,iter = inv_knothe3dcheb(Xg2,Xg3,Xg1,a2,b2,a3,b3,a1,b1,jnp.transpose(c,(1,2,0)),maxIter,alpha,tol)

            itervec = itervec.at[n].set(iter)
            gZ1 = gfun(Z1,a1,b1,nb1)
            gZ2 = gfun(Z2,a2,b2,nb2)
            gZ3 = gfun(Z3,a3,b3,nb3)
            f = jnp.einsum('abc,ia,ib,ic->i',c,gZ1,gZ2,gZ3)
            F = jnp.reshape(f,(ng1,ng2,ng3))
    return C


def Plot2DTransportInverseFit(C, np1, np2, z, mf):
    a1,b1,a2,b2,a3,b3 = 0.,mf.cell.a[0,0],0.,mf.cell.a[1,1],0.,mf.cell.a[2,2]
    N = C.shape[-1]

    # define uniform plotting grid
    xp1 = jnp.linspace(a1,b1,np1)
    xp2 = jnp.linspace(a2,b2,np2)
    X1 = jnp.repeat(jnp.reshape(xp1,(np1,1)),np2,axis=1)
    X2 = jnp.repeat(jnp.reshape(xp2,(1,np2)),np1,axis=0)
    Xp1 = jnp.reshape(X1,np1*np2)
    Xp2 = jnp.reshape(X2,np1*np2)
    Xp3 = z*jnp.ones(np1*np2)

    # compute forward and inverse transport of uniform grid
    T1,T2,T3,Jfit = flow(Xp1,Xp2,Xp3,C,a1,b1,a2,b2,a3,b3,N)
    S1,S2,S3,Jfit = inv_flow(Xp1,Xp2,Xp3,C,a1,b1,a2,b2,a3,b3,N, maxIter, alpha, tol)

    #'''
    rho = rhofun_fake(Xp1, Xp2, Xp3, mf, 0.)    
    Idx = jnp.argmax(rho)
    norm = rho[Idx]/Jfit[Idx]
    print("error in fit ", jnp.max(jnp.abs(rho - Jfit*norm)))

    #def fun(X):
    #  T1, T2, T3, _ = flowJittable(X[:1], X[1:2], X[2:], C,a1,b1,a2,b2,a3,b3,N)
    #  return jnp.asarray([T1[0], T2[0], T3[0]])
    
    #J = jax.lax.map(jacfwd(fun), jnp.hstack((Xp1.reshape(-1,1), Xp2.reshape(-1,1), Xp3.reshape(-1,1))) )
    #'''

    
    #S1,S2,S3,_ = inv_flow(Xp1,Xp2,Xp3,C,a1,b1,a2,b2,a3,b3,N,maxIter,alpha,tol)

    t1, t2, t3, _ = flow(S1, S2, S3, C, a1, b1, a2, b2, a3, b3, N)
    err1 = jnp.max(jnp.abs(t1-Xp1))#/(b1-a1)
    err2 = jnp.max(jnp.abs(t2-Xp2))#/(b2-a2)
    err3 = jnp.max(jnp.abs(t3-Xp3))#/(b3-a3)
    
    print("error in inverse", err1, err2, err3)


    # plot
    print("Forward transport image points:")
    fig, ax = plt.subplots()
    ax.scatter(T1,T2)
    #ax.set_box_aspect(1)
    plt.show()
    print("")
    print("Inverse transport image points:")
    fig, ax = plt.subplots()
    ax.scatter(S1,S2)
    #ax.set_box_aspect(1)
    plt.show()


    #'''
    ##plot 1d and compare to exact
    xp1 = jnp.linspace(a1,b1,100)
    xp2 = z*jnp.ones(100)
    xp3 = z*jnp.ones(100)
    #T1,T2,T3,Jfit = flow(xp1,xp2,xp3,C,a1,b1,a2,b2,a3,b3,N)
    S1,S2,S3,Jfit= inv_flow(xp1,xp2,xp3,C,a1,b1,a2,b2,a3,b3,N,maxIter,alpha,tol)
    rho = rhofun_fake(xp1, xp2, xp3, mf, 0.)
    norm = jnp.mean(rho/Jfit)
    plt.plot(xp1, rho, label="real")
    plt.plot(xp1, norm*Jfit, label="fit")
    plt.legend()
    plt.show()
        
    #plt.plot(S1, rho-norm*Jfit,'--')
    #plt.plot(xp1, rho-norm*Jfit, 'o-')
    plt.show()
    #'''


def Plot2DTransport(C, np1, np2, z, mf):
    a1,b1,a2,b2,a3,b3 = 0.,mf.cell.a[0,0],0.,mf.cell.a[1,1],0.,mf.cell.a[2,2]
    N = C.shape[-1]

    # define uniform plotting grid
    xp1 = jnp.linspace(a1,b1,np1)
    xp2 = jnp.linspace(a2,b2,np2)
    X1 = jnp.repeat(jnp.reshape(xp1,(np1,1)),np2,axis=1)
    X2 = jnp.repeat(jnp.reshape(xp2,(1,np2)),np1,axis=0)
    Xp1 = jnp.reshape(X1,np1*np2)
    Xp2 = jnp.reshape(X2,np1*np2)
    Xp3 = z*jnp.ones(np1*np2)

    # compute forward and inverse transport of uniform grid
    T1,T2,T3,Jfit = flow(Xp1,Xp2,Xp3,C,a1,b1,a2,b2,a3,b3,N)


    #'''
    rho = rhofun_fake(Xp1, Xp2, Xp3, mf, 0.)    
    Idx = jnp.argmax(rho)
    norm = rho[Idx]/Jfit[Idx]
    print("error in fit ", jnp.max(jnp.abs(rho - Jfit*norm)))

    def fun(X):
      T1, T2, T3, _ = flowJittable(X[:1], X[1:2], X[2:], C,a1,b1,a2,b2,a3,b3,N)
      return jnp.asarray([T1[0], T2[0], T3[0]])
    
    J = jax.lax.map(jacfwd(fun), jnp.hstack((Xp1.reshape(-1,1), Xp2.reshape(-1,1), Xp3.reshape(-1,1))) )
    #'''

    
    S1,S2,S3,_ = inv_flow(Xp1,Xp2,Xp3,C,a1,b1,a2,b2,a3,b3,N,maxIter,alpha,tol)

    t1, t2, t3, _ = flow(S1, S2, S3, C, a1, b1, a2, b2, a3, b3, N)
    err1 = jnp.max(jnp.abs(t1-Xp1))#/(b1-a1)
    err2 = jnp.max(jnp.abs(t2-Xp2))#/(b2-a2)
    err3 = jnp.max(jnp.abs(t3-Xp3))#/(b3-a3)
    
    print("error in inverse", err1, err2, err3)


    # plot
    print("Forward transport image points:")
    fig, ax = plt.subplots()
    ax.scatter(T1,T2)
    #ax.set_box_aspect(1)
    plt.show()
    print("")
    print("Inverse transport image points:")
    fig, ax = plt.subplots()
    ax.scatter(S1,S2)
    #ax.set_box_aspect(1)
    plt.show()


    #'''
    ##plot 1d and compare to exact
    xp1 = jnp.linspace(a1,b1,100)
    xp2 = z*jnp.ones(100)
    xp3 = z*jnp.ones(100)
    T1,T2,T3,Jfit = flow(xp1,xp2,xp3,C,a1,b1,a2,b2,a3,b3,N)
    S1,S2,S3,_ = inv_flow(xp1,xp2,xp3,C,a1,b1,a2,b2,a3,b3,N,maxIter,alpha,tol)
    rho = rhofun_fake(xp1, xp2, xp3, mf, 0.)
    norm = jnp.mean(rho/Jfit)
    plt.plot(xp1, rho, label="real")
    plt.plot(xp1, norm*Jfit, label="fit")
    plt.legend()
    plt.show()
    
    #plt.plot(S1, rho-norm*Jfit,'--')
    #plt.plot(xp1, rho-norm*Jfit, 'o-')
    plt.show()
    #'''
def fitFourier(mf, ng1, ng2, ng3, C):
  #a1,b1,a2,b2,a3,b3 = 0.,1,0.,1,0.,1
  a1,b1,a2,b2,a3,b3 = 0.,mf.cell.a[0,0],0.,mf.cell.a[1,1],0.,mf.cell.a[2,2]
  N = C.shape[-1]

  #'''
  dx1 = (b1-a1)/ng1
  dx2 = (b2-a2)/ng2
  dx3 = (b3-a3)/ng3
  dV = dx1*dx2*dx3
  xp1 = jnp.linspace(a1,b1-dx1,ng1) + (dx1/2)
  xp2 = jnp.linspace(a2,b2-dx2,ng2) + (dx2/2)
  xp3 = jnp.linspace(a3,b3-dx3,ng3) + (dx3/2)
  X1 = jnp.repeat(jnp.repeat(jnp.reshape(xp1,(ng1,1,1)),ng2,axis=1),ng3,axis=2)
  X2 = jnp.repeat(jnp.repeat(jnp.reshape(xp2,(1,ng2,1)),ng1,axis=0),ng3,axis=2)
  X3 = jnp.repeat(jnp.repeat(jnp.reshape(xp3,(1,1,ng3)),ng1,axis=0),ng2,axis=1)
  Xp1 = jnp.reshape(X1,ng1*ng2*ng3)
  Xp2 = jnp.reshape(X2,ng1*ng2*ng3)
  Xp3 = jnp.reshape(X3,ng1*ng2*ng3)
  #'''
  
  # Compute forward and inverse flow on equispaced grid
  #Tp1,Tp2,Tp3,_ = flow(Xp1,Xp2,Xp3,C,a1,b1,a2,b2,a3,b3,N)
  Sp1,Sp2,Sp3,_ = inv_flow(Xp1,Xp2,Xp3,C,a1,b1,a2,b2,a3,b3,N,maxIter,alpha,tol)
  #Tp1 = jnp.reshape(Tp1-Xp1,(ng1,ng2,ng3))
  #Tp2 = jnp.reshape(Tp2-Xp2,(ng1,ng2,ng3))
  #Tp3 = jnp.reshape(Tp3-Xp3,(ng1,ng2,ng3))
  Sp1 = jnp.reshape(Sp1-Xp1,(ng1,ng2,ng3))
  Sp2 = jnp.reshape(Sp2-Xp2,(ng1,ng2,ng3))
  Sp3 = jnp.reshape(Sp3-Xp3,(ng1,ng2,ng3))

  nf1 = (ng1-1)//2
  nf2 = (ng2-1)//2
  nf3 = (ng3-1)//2

  # define Fourier mode evaluator
  def ffun(x,a,b,nf):
    nx = x.shape[0]
    u = jnp.reshape( 2*jnp.pi*(x - (a+b)/2)/(b-a), (nx,1) )
    n = jnp.reshape( jnp.arange(-nf,nf+1), (1,2*nf+1) )
    f = jnp.exp(1j*n*u) / jnp.sqrt(b-a)
    return f

  # define Fourier fit function
  @jit
  def fourierfit(F,f1,f2,f3,dV):
    c = jnp.einsum('ijk,ia,jb,kc->abc',F,jnp.conj(f1),jnp.conj(f2),jnp.conj(f3)) * dV
    return c

  # evaluate Fourier modes on quadrature grid
  f1 = ffun(xp1,a1,b1,nf1)
  f2 = ffun(xp2,a2,b2,nf2)
  f3 = ffun(xp3,a3,b3,nf3)

  #import pdb
  #pdb.set_trace()

  # Fit the component functions with Fourier modes
  #F_T1 = fourierfit(Tp1,f1,f2,f3,dV)
  #F_T2 = fourierfit(Tp2,f1,f2,f3,dV)
  #F_T3 = fourierfit(Tp3,f1,f2,f3,dV)
  F_S1 = fourierfit(Sp1,f1,f2,f3,dV)
  F_S2 = fourierfit(Sp2,f1,f2,f3,dV)
  F_S3 = fourierfit(Sp3,f1,f2,f3,dV)
  return F_S1, F_S2, F_S3