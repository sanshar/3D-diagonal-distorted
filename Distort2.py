import os
#import numpy as jnp
import matplotlib.pyplot as plt
#from jax import lax
from jax import numpy as jnp
from jax import scipy as jsp
from jax import value_and_grad, vmap, lax, jit, random
#os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
from scipy.optimize import bisect
from scipy.optimize import fsolve
import numpy

@jit
def flow2(grid, C, a1, b1, a2, b2, a3, b3):
  f = b1 / (jnp.sinh(b1)*0.8)
  gridnew = jnp.arcsinh(grid/0.8/f)
  return gridnew 

#@jit
def invflow2(grid, C, a1, b1, a2, b2, a3, b3):
  f = b1 / (jnp.sinh(b1)*0.8)
  gridnew = (jnp.sinh(grid) * 0.8 * f) 
  return gridnew #jnp.einsum('gi,g->gi', grid, rnew/r)

@jit
def flow3(grid, C, a1, b1, a2, b2, a3, b3):
  return jnp.hstack((grid, jnp.ones((grid.shape[0],)))) #jnp.einsum('gi,g->gi', grid, rnew/r)
  #return grid*rnew/r

#@jit
def invflow3(grid, C, a1, b1, a2, b2, a3, b3):
  return grid #jnp.einsum('gi,g->gi', grid, rnew/r)

  r = jnp.sum(grid**2, axis=1)**0.5
  rnew = jnp.sinh(r)*0.2
  return jnp.einsum('gi,g->gi', grid, rnew/r)

@jit
def f(x):
  return jsp.special.erf(x*3.2**0.5) * (jnp.pi/3.2)**0.5/2 + 0.00001*x

#@jit
def finv(x, a, b):
  xout = numpy.zeros((x.shape[0],x.shape[1]))
  for i, xi in enumerate(x):
    for j, xij in enumerate(xi):
      xout[i,j] = bisect(lambda r : f(r) - xij, a, b)
  return xout

#@jit
def flow4(grid, C, a1, b1, a2, b2, a3, b3):
  gridnew = a1 + (f(grid) - f(a1))/(f(b1)-f(a1)) * (b1-a1)
  return jnp.hstack((gridnew, jnp.prod(jnp.exp(-3.2*grid**2)+0.00001)* (6/(f(b1)-f(a1)))**3 )) 

#@jit
def invflow4(grid, C, a1, b1, a2, b2, a3, b3):
  gridnew = finv((grid-a1)*(f(b1)-f(a1))/6.+f(a1), a1, b1)
  return gridnew #jnp.einsum('gi,g->gi', grid, rnew/r)


alpha = 1.6
@jit
def f1(x):
  return jsp.special.erf(x*alpha**0.5) * 3.0001

#@jit
def finv1(x, a, b):
  return jsp.special.erfinv((x)/3.0001)/(alpha**0.5)

#@jit
def flow5(grid, C, a1, b1, a2, b2, a3, b3):
  gridnew = f1(grid)
  return jnp.hstack((gridnew, jnp.prod(jnp.exp(-alpha*grid**2)) )) 

#@jit
def invflow5(grid, C, a1, b1, a2, b2, a3, b3):
  gridnew = finv1(grid, a1, b1)
  return gridnew #jnp.einsum('gi,g->gi', grid, rnew/r)


def flow6(grid, C, a1, b1, a2, b2, a3, b3):
  ngrid = grid.shape[0]

  r2 = jnp.sum(grid**2, axis=1)
  Jac = jnp.tile(jnp.eye(3), (ngrid,1)).reshape(ngrid,3,3)
  Jac = jnp.einsum('gij,g->gij',Jac, (1-0.9*jnp.exp(-r2)))
  Jac += 2.*jnp.einsum('gi,gj,g->gij', grid, grid, 0.9*jnp.exp(-r2))
  
  return Jac

def invflow6(grid, C, a1, b1, a2, b2, a3, b3):
  r2 = jnp.sum(grid**2, axis=1)
  gridout = grid - 0.9*jnp.einsum('gi,g->gi',grid , jnp.exp(-r2))
  return gridout

@jit
def chebfit(F,g1,g2,g3,S):
  c = jnp.einsum('ijk,ia->ajk',F,g1)
  c = jnp.einsum('ijk,jb->ibk',c,g2)
  c = jnp.einsum('ijk,kc->ijc',c,g3) / S
  return c

@jit
def gfun(x,a,b,nrange):
  u = 2*(x - (a+b)/2)/(b-a)
  g = jnp.cos(nrange*jnp.arccos(u))
  return g

@jit
def Gtildefun(x,a,b,nb, nbrange):

  u = 2*(x - (a+b)/2)/(b-a)
  G0 = u
  G1 = u**2/2

  g = gfun(x,a,b,nbrange)
  gplus = g[3:]
  gminus = g[1:-2]
  nvec = jnp.arange(2,nbrange.shape[0]-1)
  G2 = gplus/(2*(nvec+1)) - gminus/(2*(nvec-1))

  G = jnp.hstack((G0,G1,G2))
  return G

@jit
def Gfun(x,a,b,nb,nbrange):
  Gtilde = Gtildefun(x,a,b,nb,nbrange)
  Gtilde0 = Gtildefun(a*jnp.ones(1),a,b,nb, nbrange)
  G = (b-a)*(Gtilde - Gtilde0)/2
  return G

# define Knothe transport function
@jit
def knothe3dcheb(x,a1,b1,a2,b2,a3,b3,c):
  T = 0.*x
  
  L1 = b1-a1
  L2 = b2-a2
  L3 = b3-a3
  nb1 = c.shape[0]
  nb2 = c.shape[1]
  nb3 = c.shape[2]

  nb1range = jnp.arange(nb1+1)
  nb2range = jnp.arange(nb2+1)
  nb3range = jnp.arange(nb3+1)
  m1 = jnp.ravel( Gfun(b1*jnp.ones(1),a1,b1,nb1, nb1range) )
  m2 = jnp.ravel( Gfun(b2*jnp.ones(1),a2,b2,nb2, nb2range) )
  m3 = jnp.ravel( Gfun(b3*jnp.ones(1),a3,b3,nb3, nb3range) )

  M = jnp.einsum('abc,a,b,c',c,m1,m2,m3)
  cc = c/M

  nb1range = jnp.arange(nb1)
  nb2range = jnp.arange(nb2)
  nb3range = jnp.arange(nb3)
  g1 = gfun(x[0],a1,b1,nb1range)
  g2 = gfun(x[1],a2,b2,nb2range)
  g3 = gfun(x[2],a3,b3,nb3range)

  nb1range = jnp.arange(nb1+1)
  nb2range = jnp.arange(nb2+1)
  nb3range = jnp.arange(nb3+1)
  G1 = Gfun(x[0],a1,b1,nb1, nb1range)
  G2 = Gfun(x[1],a2,b2,nb2, nb2range)
  G3 = Gfun(x[2],a3,b3,nb3, nb3range)

  rho1 = jnp.einsum('abc,a,b,c',cc,g1,m2,m3)
  rho2 = jnp.einsum('abc,a,b,c',cc,g1,g2,m3)
  rho  = jnp.einsum('abc,a,b,c',cc,g1,g2,g3)

  A, B, C = jnp.einsum('abc,a,b,c',cc,G1,m2,m3), jnp.einsum('abc,a,b,c',cc,g1,G2,m3), jnp.einsum('abc,a,b,c',cc,g1,g2,G3)
  T0 = a1 + L1*A
  T1 = a2 + L2*B/rho1
  T2 = a3 + L3*C/rho2

  return jnp.hstack((T0,T1,T2)), rho*L1*L2*L3

@jit
def flow(x,C,a1,b1,a2,b2,a3,b3):
  N = C.shape[3]
  T = x
  J = 1.
  for n in range(N//3):
    #c = C[:,:,:,n]
    #if jnp.mod(n,3)==0:
    T,rhofac = knothe3dcheb(T,a1,b1,a2,b2,a3,b3,C[:,:,:,3*n])
    J = J * rhofac
    #print(n, T, rhofac  )
    
    T = jnp.hstack((T[2], T[0], T[1]))
    T,rhofac = knothe3dcheb(T,a3,b3,a1,b1,a2,b2,jnp.transpose(C[:,:,:,3*n+1],(2,0,1)))
    J = J * rhofac
    #print(n, T, J, rhofac)
    
    #T = jnp.hstack((T[1], T[2], T[0]))
    T = jnp.hstack((T[2], T[0], T[1]))
    T,rhofac = knothe3dcheb(T,a2,b2,a3,b3,a1,b1,jnp.transpose(C[:,:,:,3*n+2],(1,2,0)))
    J = J * rhofac
    #print(n, T, J, rhofac)

    T = jnp.hstack((T[2], T[0], T[1]))
    #J = J / jnp.sum(J)

  return jnp.hstack((T[0], T[1], T[2], J))
flow_vmap = vmap(flow, in_axes=(0,None,None,None,None,None,None,None))

def knothe3dcheb2(x1,x2,x3,a1,b1,a2,b2,a3,b3,c):

  L1 = b1-a1
  L2 = b2-a2
  L3 = b3-a3
  nb1 = c.shape[0]
  nb2 = c.shape[1]
  nb3 = c.shape[2]
  nb1range = jnp.arange(nb1+1)
  nb2range = jnp.arange(nb2+1)
  nb3range = jnp.arange(nb3+1)
  m1 = jnp.ravel( Gfun(b1*jnp.ones(1),a1,b1,nb1, nb1range) )
  m2 = jnp.ravel( Gfun(b2*jnp.ones(1),a2,b2,nb2, nb2range) )
  m3 = jnp.ravel( Gfun(b3*jnp.ones(1),a3,b3,nb3, nb3range) )

  M = jnp.einsum('abc,a,b,c',c,m1,m2,m3)
  cc = c/M

  nb1range = jnp.arange(nb1)
  nb2range = jnp.arange(nb2)
  nb3range = jnp.arange(nb3)
  g1 = gfun(x1,a1,b1,nb1range)
  g2 = gfun(x2,a2,b2,nb2range)
  g3 = gfun(x3,a3,b3,nb3range)

  nb1range = jnp.arange(nb1+1)
  nb2range = jnp.arange(nb2+1)
  nb3range = jnp.arange(nb3+1)
  G1 = Gfun(x1,a1,b1,nb1, nb1range)
  G2 = Gfun(x2,a2,b2,nb2, nb2range)
  G3 = Gfun(x3,a3,b3,nb3, nb3range)

  rho1 = jnp.einsum('abc,a,b,c',cc,g1,m2,m3)
  rho2 = jnp.einsum('abc,a,b,c',cc,g1,g2,m3)
  rho  = jnp.einsum('abc,a,b,c',cc,g1,g2,g3)

  T1 = a1 + L1*jnp.einsum('abc,ia,b,c->i',cc,G1,m2,m3)
  T2 = a2 + L2*jnp.einsum('abc,ia,ib,c->i',cc,g1,G2,m3)/rho1
  T3 = a3 + L3*jnp.einsum('abc,ia,ib,ic->i',cc,g1,g2,G3)/rho2

  return T1,T2,T3,rho


def create_cond_and_body(y1,y2,y3,a1,b1,a2,b2,a3,b3,c,maxIter,alpha,tol):

  #@jit
  def body_fun(carry):
    x1,x2,x3,T1,T2,T3,rho,iter = carry
    x1 = x1 - alpha*(T1-y1)
    x2 = x2 - alpha*(T2-y2)
    x3 = x3 - alpha*(T3-y3)
    T1,T2,T3,rho = knothe3dcheb2(x1,x2,x3,a1,b1,a2,b2,a3,b3,c)
    return (x1,x2,x3,T1,T2,T3,rho,iter+1)

  #@jit
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

def inv_knothe3dcheb(y1,y2,y3,a1,b1,a2,b2,a3,b3,c,maxIter,alpha,tol):
  body_fun, cond_fun = create_cond_and_body(y1,y2,y3,a1,b1,a2,b2,a3,b3,c,maxIter,alpha,tol)
  x1 = y1
  x2 = y2
  x3 = y3
  T1, T2, T3, rho = knothe3dcheb2(x1,x2,x3,a1,b1,a2,b2,a3,b3,c)
  init_val = (x1,x2,x3,T1,T2,T3,rho,0)
  x1,x2,x3,T1,T2,T3,rho,iter = lax.while_loop(cond_fun, body_fun, init_val)
  return x1,x2,x3,1/rho,iter

def inv_flow(x1,x2,x3,C,a1,b1,a2,b2,a3,b3,maxIter,alpha,tol):
  T1 = x1
  T2 = x2
  T3 = x3
  N = C.shape[3]
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
    print(n, T1, T2, T3, rhoinvfac)
    Jinv = Jinv * rhoinvfac
    Jinv = Jinv / jnp.sum(Jinv)
  return T1,T2,T3,Jinv


if __name__ == '__main__':
    # bounding box
    a1 = -3.0
    b1 = 4.0
    a2 = -4.0
    b2 = 3.5
    a3 = -2.0
    b3 = 2.0

    # number of flow steps
    N = 45

    # number of fitting Chebyshev polynomials per dimension
    nb1 = 20  # dimension 1
    nb2 = 20  # dimension 2
    nb3 = 20  # dimension 3

    # hyperparameters for fixed-point iteration computation of inverse flow
    maxIter = 1000 # maximum number of iterations
    alpha = 0.8 # mixing parameter
    tol = 1e-12 # convergence tolerance

    # plotting grid size
    jnp1 = 20
    jnp2 = 25

    # z coordinate
    z = -0.0


    # plotting grid size
    jnp1 = 20
    jnp2 = 25

    # z coordinate
    z = -1

    # define uniform plotting grid
    xp1 = jnp.linspace(a1,b1,jnp1)
    xp2 = jnp.linspace(a2,b2,jnp2)
    X1 = jnp.repeat(jnp.reshape(xp1,(jnp1,1)),jnp2,axis=1)
    X2 = jnp.repeat(jnp.reshape(xp2,(1,jnp2)),jnp1,axis=0)
    Xp1 = jnp.reshape(X1,jnp1*jnp2)
    Xp2 = jnp.reshape(X2,jnp1*jnp2)
    Xp3 = z*jnp.ones(jnp1*jnp2)

    C = jnp.fromfile("cheb.bin", dtype=jnp.float64).reshape(nb1, nb2, nb3, -1)
    # compute forward and inverse transport of uniform grid 
    T1,T2,T3,J = flow(Xp1,Xp2,Xp3,C,a1,b1,a2,b2,a3,b3)

    print("Forward transport image points:")
    fig, ax = plt.subplots()
    ax.scatter(T1,T2)
    ax.set_box_aspect(1)
    plt.show()

    plt.tricontourf(Xp1,Xp2,J,100)
    plt.show()
