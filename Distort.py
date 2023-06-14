import os
import numpy as np
import matplotlib.pyplot as plt

def chebfit(F,g1,g2,g3,S):
  c = np.einsum('ijk,ia->ajk',F,g1)
  c = np.einsum('ijk,jb->ibk',c,g2)
  c = np.einsum('ijk,kc->ijc',c,g3) / S
  return c

def gfun(x,a,b,nb):
  nx = x.shape[0]
  xc = np.clip(x,a,b)
  u = np.reshape( 2*(xc - (a+b)/2)/(b-a), (nx,1) )
  n = np.reshape( np.arange(nb), (1,nb) )
  g = np.cos(n*np.arccos(u))
  return g

def Gtildefun(x,a,b,nb):

  nx = x.shape[0]
  u = np.reshape( 2*(x - (a+b)/2)/(b-a), (nx,1) )
  G0 = u
  G1 = u**2/2

  g = gfun(x,a,b,nb+1)
  gplus = g[:,3:nb+1]
  gminus = g[:,1:nb-1]
  nvec = np.reshape(np.arange(2,nb),(1,nb-2))
  G2 = gplus/(2*(nvec+1)) - gminus/(2*(nvec-1))

  G = np.hstack((G0,G1,G2))
  return G

def Gfun(x,a,b,nb):
  Gtilde = Gtildefun(x,a,b,nb)
  Gtilde0 = Gtildefun(a*np.ones(1),a,b,nb)
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
  m1 = np.ravel( Gfun(b1*np.ones(1),a1,b1,nb1) )
  m2 = np.ravel( Gfun(b2*np.ones(1),a2,b2,nb2) )
  m3 = np.ravel( Gfun(b3*np.ones(1),a3,b3,nb3) )

  M = np.einsum('abc,a,b,c',c,m1,m2,m3)
  cc = c/M

  g1 = gfun(x1,a1,b1,nb1)
  g2 = gfun(x2,a2,b2,nb2)
  g3 = gfun(x3,a3,b3,nb3)
  G1 = Gfun(x1,a1,b1,nb1)
  G2 = Gfun(x2,a2,b2,nb2)
  G3 = Gfun(x3,a3,b3,nb3)

  rho1 = np.einsum('abc,ia,b,c->i',cc,g1,m2,m3)
  rho2 = np.einsum('abc,ia,ib,c->i',cc,g1,g2,m3)
  rho = np.einsum('abc,ia,ib,ic->i',cc,g1,g2,g3)

  T1 = a1 + L1*np.einsum('abc,ia,b,c->i',cc,G1,m2,m3)
  T2 = a2 + L2*np.einsum('abc,ia,ib,c->i',cc,g1,G2,m3)/rho1
  T3 = a3 + L3*np.einsum('abc,ia,ib,ic->i',cc,g1,g2,G3)/rho2

  return T1,T2,T3,rho

def flow(x1,x2,x3,C,a1,b1,a2,b2,a3,b3):
  N = C.shape[3]
  T1 = x1
  T2 = x2
  T3 = x3
  p = x1.shape[0]
  J = np.ones(p)/p
  for n in range(N):
    c = C[:,:,:,n]
    if np.mod(n,3)==0:
      T1,T2,T3,rhofac = knothe3dcheb(T1,T2,T3,a1,b1,a2,b2,a3,b3,c)
    elif np.mod(n,3)==1:
      T3,T1,T2,rhofac = knothe3dcheb(T3,T1,T2,a3,b3,a1,b1,a2,b2,np.transpose(c,(2,0,1)))
    else:
      T2,T3,T1,rhofac = knothe3dcheb(T2,T3,T1,a2,b2,a3,b3,a1,b1,np.transpose(c,(1,2,0)))
    J = J * rhofac
    J = J / np.sum(J)
  return T1,T2,T3,J

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
    np1 = 20
    np2 = 25

    # z coordinate
    z = -0.0


    # plotting grid size
    np1 = 20
    np2 = 25

    # z coordinate
    z = -1

    # define uniform plotting grid
    xp1 = np.linspace(a1,b1,np1)
    xp2 = np.linspace(a2,b2,np2)
    X1 = np.repeat(np.reshape(xp1,(np1,1)),np2,axis=1)
    X2 = np.repeat(np.reshape(xp2,(1,np2)),np1,axis=0)
    Xp1 = np.reshape(X1,np1*np2)
    Xp2 = np.reshape(X2,np1*np2)
    Xp3 = z*np.ones(np1*np2)

    C = np.fromfile("cheb.bin", dtype=np.float64).reshape(nb1, nb2, nb3, -1)
    # compute forward and inverse transport of uniform grid 
    T1,T2,T3,J = flow(Xp1,Xp2,Xp3,C,a1,b1,a2,b2,a3,b3)

    print("Forward transport image points:")
    fig, ax = plt.subplots()
    ax.scatter(T1,T2)
    ax.set_box_aspect(1)
    plt.show()

    plt.tricontourf(Xp1,Xp2,J,100)
    plt.show()
