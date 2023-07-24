import os
import numpy as jnp
import matplotlib.pyplot as plt
from jax import lax, jit
from jax import numpy as jnp
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

@jit
def chebfit(F,g1,g2,g3,S):
  c = jnp.einsum('ijk,ia->ajk',F,g1)
  c = jnp.einsum('ijk,jb->ibk',c,g2)
  c = jnp.einsum('ijk,kc->ijc',c,g3) / S
  return c

#@jit
def gfun(x,a,b,nb):
  nx = x.shape[0]
  xc = jnp.clip(x,a,b)
  u = jnp.reshape( 2*(xc - (a+b)/2)/(b-a), (nx,1) )
  n = jnp.reshape( jnp.arange(nb), (1,nb) )
  g = jnp.cos(n*jnp.arccos(u))
  return g

#@jit
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

#@jit
def Gfun(x,a,b,nb):
  Gtilde = Gtildefun(x,a,b,nb)
  Gtilde0 = Gtildefun(a*jnp.ones(1),a,b,nb)
  G = (b-a)*(Gtilde - Gtilde0)/2
  return G

# define Knothe transport function
#@jit
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

  return T1,T2,T3,rho

#@jit
def flow(x1,x2,x3,C,a1,b1,a2,b2,a3,b3):
  N = C.shape[3]
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
    #print(n, T1, T2, T3, J, rhofac)
  #J = J / jnp.sum(J)
  return T1,T2,T3,J

def create_cond_and_body(y1,y2,y3,a1,b1,a2,b2,a3,b3,c,maxIter,alpha,tol):

  #@jit
  def body_fun(carry):
    x1,x2,x3,T1,T2,T3,rho,iter = carry
    x1 = x1 - alpha*(T1-y1)
    x2 = x2 - alpha*(T2-y2)
    x3 = x3 - alpha*(T3-y3)
    T1,T2,T3,rho = knothe3dcheb(x1,x2,x3,a1,b1,a2,b2,a3,b3,c)
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
  T1, T2, T3, rho = knothe3dcheb(x1,x2,x3,a1,b1,a2,b2,a3,b3,c)
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
