from jax import numpy as jnp
from jax import scipy as jsp
from jax import value_and_grad, vmap, lax, jit, random

import jax, jaxopt
import scipy.special
import numpy as np
import matplotlib.pyplot as plt

maxIter = 1000 # maximum number of iterations
alpha = 0.3 # mixing parameter
tol = 1e-15 # convergence tolerance

@jit
def gfun(alpha, c, x):
    nx , nc = x.shape[0], c.shape[0]
    X = jnp.repeat( jnp.reshape(x, (nx, 1)), nc, axis=1)
    return jnp.exp(-alpha * (X-c)**2 )

@jit
def Gfun(alpha, c, x, a):
    nx , nc = x.shape[0], c.shape[0]
    X = jnp.repeat( jnp.reshape(x, (nx, 1)), nc, axis=1)
    alpha = alpha + 1.e-12  ## for zero alpha there are numerical issues
    X = X + 1.e-12
    return (jnp.pi/alpha)**0.5 /2. * (jsp.special.erf(alpha**0.5 * (X-c)) - jsp.special.erf(alpha**0.5 * (a-c)) )

@jit
def flow(x1, x2, x3, C, a1, b1, a2, b2, a3, b3):
    L1 = b1-a1
    L2 = b2-a2
    L3 = b3-a3

    Ngauss = C.shape[0]//3
    coeff, center, alpha = C[:Ngauss], C[Ngauss:2*Ngauss], C[2*Ngauss:] 

    m1 = Gfun(alpha, center, b1*jnp.ones((1,)), a1)[0]
    m2 = Gfun(alpha, center, b2*jnp.ones((1,)), a2)[0]
    m3 = Gfun(alpha, center, b3*jnp.ones((1,)), a3)[0]

    #normalize the coefficient
    M = jnp.einsum('a,a,a,a', coeff, m1, m2, m3)
    cc = coeff/M

    g1 = gfun(alpha, center, x1)
    g2 = gfun(alpha, center, x2)
    g3 = gfun(alpha, center, x3)
    G1 = Gfun(alpha, center, x1, a1)
    G2 = Gfun(alpha, center, x2, a2)
    G3 = Gfun(alpha, center, x3, a3)

    #print(g1, G1, cc)
    rho1 = jnp.einsum('a,ia,a,a->i',cc,g1,m2,m3)
    rho2 = jnp.einsum('a,ia,ia,a->i',cc,g1,g2,m3)
    rho  = jnp.einsum('a,ia,ia,ia->i',cc,g1,g2,g3)

    T1 = a1 + L1*jnp.einsum('a,ia,a,a->i',cc,G1,m2,m3)
    T2 = a2 + L2*jnp.einsum('a,ia,ia,a->i',cc,g1,G2,m3)/rho1
    T3 = a3 + L3*jnp.einsum('a,ia,ia,ia->i',cc,g1,g2,G3)/rho2


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

    T1, T2, T3, rho = flow(x1, x2, x3, c, a1, b1, a2, b2, a3, b3)
    return jnp.linalg.norm((allT - jnp.hstack((T1, T2, T3)) ))

'''
def invflow_explicit(y1, y2, y3, c, a1, b1, a2, b2, a3, b3):
    L1 = b1-a1
    L2 = b2-a2
    L3 = b3-a3

    Ngauss = C.shape[0]//3
    coeff, center, alpha = C[:Ngauss], C[Ngauss:2*Ngauss], C[2*Ngauss:] 

    m1 = Gfun(alpha, center, b1*jnp.ones((1,)), a1)[0]
    m2 = Gfun(alpha, center, b2*jnp.ones((1,)), a2)[0]
    m3 = Gfun(alpha, center, b3*jnp.ones((1,)), a3)[0]

    #normalize the coefficient
    M = jnp.einsum('a,a,a,a', coeff, m1, m2, m3)
    cc = coeff/M

    lambda fun x : a1 + L1*jnp.einsum('a,ia,a,a->i',cc,Gfun(alpha, center, x, a1),m2,m3)
    S2 = (y1 - a1)/L1 
'''
@jit
def invflow(y1,y2,y3,c,a1,b1,a2,b2,a3,b3,maxIter,alpha,tol):
    body_fun, cond_fun = create_cond_and_body(y1,y2,y3,a1,b1,a2,b2,a3,b3,c,maxIter,alpha,tol)
    x1 = y1
    x2 = y2
    x3 = y3
    T1, T2, T3, rho = flow(x1,x2,x3,c,a1,b1,a2,b2,a3,b3)
    init_val = (x1,x2,x3,T1,T2,T3,rho,0)

    solver = jaxopt.GradientDescent(fun=errFun, maxiter=maxIter)
    #solver = jaxopt.NonlinearCG(fun=errFun, method="polak-ribiere", maxiter=maxIter)

    res = solver.run(jnp.hstack((T1,T2,T3)), jnp.hstack((y1,y2,y3)), c,a1,b1,a2,b2,a3,b3)      

    N = y1.shape[0]
    allS = res.params
    return allS[:N], allS[N:2*N], allS[2*N:], 1./rho


def create_cond_and_body(y1,y2,y3,a1,b1,a2,b2,a3,b3,c,maxIter,alpha,tol):

  @jit
  def body_fun(carry):
    x1,x2,x3,T1,T2,T3,rho,iter = carry
    x1 = x1 - alpha*(T1-y1)
    x2 = x2 - alpha*(T2-y2)
    x3 = x3 - alpha*(T3-y3)
    T1,T2,T3,rho = flow(x1,x2,x3,c,a1,b1,a2,b2,a3,b3)
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


if __name__ =="__main__":
    C = jnp.asarray([  2.3099184,   6.475843,   12.867022, 0.05, 5., 5., 5., 5.,   1.6527886,  26.1984,    180.28716,0.  ])
    #C = jnp.asarray([  2.3099184,   6.475843,   12.867022, 0.0005, 0., 0., 0., 0.,   1.6527886,  26.1984,    180.28716,0.  ])
    #C = jnp.asarray([  2.3099184, 0.,   10.6527886])

    


    a1, b1 = -5, 5
    a2, b2 = -5, 5
    a3, b3 = -5, 5

    #'''
    np1, np2, np3 = 25, 25, 25

    x1 = jnp.linspace(a1, b1, np1)
    x2 = jnp.linspace(a2, b2, np2)

    X1 = jnp.repeat(jnp.reshape(x1,(np1,1)),np2,axis=1)
    X2 = jnp.repeat(jnp.reshape(x2,(1,np2)),np1,axis=0)
    Xp1 = jnp.reshape(X1,np1*np2)
    Xp2 = jnp.reshape(X2,np1*np2)
    Xp3 = 0.*jnp.ones(np1*np2)

    #print(flow(0*jnp.ones((1,)), b2*jnp.ones((1,)), 0*jnp.ones((1,)), C, a1, b1, a2, b2, a3, b3))
    #T1, T2, T3, rho = flow(x1, x2, x2, C, a1, b1, a2, b2, a3, b3)
    #exit(0)
    T1,T2,T3,_ = flow(Xp3,Xp2,Xp1,C,a1,b1,a2,b2,a3,b3)
    S1,S2,S3,_ = invflow(Xp3,Xp2,Xp1,C,a1,b1,a2,b2,a3,b3,maxIter,alpha,tol)

    t1, t2, t3, _ = flow(S1, S2, S3, C, a1, b1, a2, b2, a3, b3)
    err1 = jnp.max(jnp.abs(t1-Xp3))/(b1-a1)
    err2 = jnp.max(jnp.abs(t2-Xp2))/(b2-a2)
    err3 = jnp.max(jnp.abs(t3-Xp1))/(b3-a3)
    
    print(err1, err2, err3)

    #print(Xp1)
    #print(Xp2)
    #print(T1)
    #print(T2)
    fig, ax = plt.subplots()
    ax.scatter(T2,T3)
    #ax.set_box_aspect(1)
    plt.show()

    #print(f"solution required {iter} iterations with final tol = {tol}")
    fig, ax = plt.subplots()
    ax.scatter(S2,S3)
    #ax.set_box_aspect(1)
    plt.show()

    #'''
    N = 4
    xs = np.linspace(a1, b1, 1001)

    m1 = Gfun(C[2*N:], C[N:2*N], b1*jnp.ones((1,)), a1)[0]
    m2 = Gfun(C[2*N:], C[N:2*N], b1*jnp.ones((1,)), a1)[0]
    m3 = Gfun(C[2*N:], C[N:2*N], b1*jnp.ones((1,)), a1)[0]

    #normalize the coefficient
    M = jnp.einsum('a,a,a,a', C[:4], m1, m2, m3)
    cc = C[:4]/M


    g1 = gfun(C[2*N:], C[N:2*N], xs)
    g2 = gfun(C[2*N:], C[N:2*N], xs)
    g3 = gfun(C[2*N:], C[N:2*N], xs)
    G1 = Gfun(C[2*N:], C[N:2*N], xs, a1)
    G2 = Gfun(C[2*N:], C[N:2*N], xs, a2)
    G3 = Gfun(C[2*N:], C[N:2*N], xs, a3)

    #print(g1, G1, cc)
    rho1 = jnp.einsum('a,ia,a,a->i',cc,g1,m2,m3)
    rho2 = jnp.einsum('a,a,ia,a->i',cc,g1[500],g2,m3)
    rho  = jnp.einsum('a,a,a,ia->i',cc,g1[500],g2[500],g3)


    plt.plot(xs, rho1, label="rho1")
    plt.plot(xs, (rho2), label="rho2")
    plt.plot(xs, (rho), label="rho3")
    plt.legend()
    plt.show()
    import pdb
    pdb.set_trace()