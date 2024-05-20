import os
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=6'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
#os.environ['JAX_DISABLE_JIT'] = 'True'
from jax import numpy as jnp
from jax import scipy as jsp
from jax import value_and_grad, vmap, lax, jit, random, jacfwd, jacrev

import pyscf, time, scipy
import jax, jaxopt
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from pyscf.pbc import gto as pgto


#implements log of poch-hammer symbol
@jit 
def poch(x,n):
    return jnp.sum(jnp.log(jnp.arange(x, x+n-0.1, 1)))

#hypergeometric series 2F2([0.5,0.5], [1.5, 1.5], x) but it does not converge for large negative values of 
#x because the power series is oscillating
@jit
def hyp22(x):
    sgn = jnp.sign(x)
    f = lambda n, val : val +  sgn**n * jnp.exp(2.*poch(0.5,n) - 2.*poch(1.5,n) + n * jnp.log(abs(x)) - jsp.special.gammaln(n+1))

    val = jax.lax.fori_loop(0, 100, f, 0.)
    return val

#integrates the erf[Ax]/x type deformation but getting numerically stable results are difficult
@jit 
def integralRho_Erf(params, x, Zi, pos):
    a, b, shift = params[0], params[1], params[2]
    R = (x - pos) + 1.e-8

    integral = hyp22(- (Zi*R/a)**2 ) *2 * (Zi*R/a)/jnp.pi**0.5  -\
        hyp22(- (R/b)**2 ) *2 * (R/b)/jnp.pi**0.5 + shift*R 
    return integral

@jit 
def integralRho_Gauss(params, x, Zi, pos):
    N = params.shape[0]//2
    X = jnp.repeat(jnp.reshape(x, (x.shape[0],1)),N, axis=1)
    R = (X - pos) + 1.e-8

    integral = params[:N]*(jsp.special.erf((params[N:]+1.e-8)**0.5 * R) * (np.pi/(params[N:]+1.e-8))**0.5 / 2.)
    return jnp.einsum('ij->i', integral)

@jit
def rho_erf(params, x, Zi, pos, a1, b1):
    a, b, shift = params[0], params[1], params[2]

    X = jnp.repeat(jnp.reshape(x, (x.shape[0],1)),Zi.shape[0], axis=1)
    R = (X - pos) + 1.e-8

    d = (jsp.special.erf(R * Zi / a) - jsp.special.erf( R / b))/R 

    integral = integralRho_Erf(params, b1, Zi, pos) - integralRho_Erf(params, a1, Zi, pos)

    return (jnp.einsum('ij->i', d) + shift)/integral, params/integral

##single atom
@jit
def rho_Gauss(params, x, Zi, pos, a1, b1):
    N = params.shape[0]//2

    X = jnp.repeat(jnp.reshape(x, (x.shape[0],1)),N, axis=1)
    R = (X - pos[0])
    fval = params[:N] * jnp.exp(-params[N:] * R**2) 

    integral = integralRho_Gauss(params, b1*jnp.ones((1,)), Zi, pos) - integralRho_Gauss(params, a1*jnp.ones((1,)), Zi, pos)
    normalizedParams = 1.*params
    normalizedParams.at[:N].set(params[:N]/integral)
    return jnp.einsum('ij->i', fval)/integral, normalizedParams

@jit 
def Tmap(params, x, Zi, center, a1, b1):
    L1 = b1 - a1
    T0 = integralRho_Gauss(params, a1*jnp.ones((1,)), Zi*jnp.ones((1,)), center*jnp.ones((1,)))
    T1 = integralRho_Gauss(params, b1*jnp.ones((1,)), Zi*jnp.ones((1,)), center*jnp.ones((1,)))
    T = a1 + L1 * (integralRho_Gauss(params, x, Zi*jnp.ones((1,)), center*jnp.ones((1,))) - T0)/(T1-T0)
    return T
    

@jit
def invMap(params, y, Zi, center, a1, b1):

    tol, maxIter, alpha = 1.e-7, 6000, 0.05

    @jit
    def cond_fun(carry):
        x, T, iter = carry
        err = jnp.max(jnp.abs(x-T))/(b1-a1)
        return (err > tol) * (iter < maxIter)

    @jit 
    def body_fun(carry):
        x, T, iter = carry
        x = x - alpha * (T - y)
        T = Tmap(params, x, Zi, center, a1, b1)
        return (x, T, iter+1)

    @jit
    def body_scan(carry, r):
        carry = body_fun(carry)
        return carry, r

    x = y
    init_val = (x, Tmap(params, x, Zi, center, a1, b1), 0)
    carry, r = jax.lax.scan(body_scan, init_val, None, maxIter)
    #carry = lax.while_loop(cond_fun, body_fun, init_val)
    return carry[0], carry[2]

    #solver = jaxopt.GradientDescent(fun = fun, maxiter = 3000)
    #solver = jaxopt.LBFGS(fun = fun, maxiter = 3000)
    #res = solver.run(x, x )

    return res.params, Tmap(params, res.params, Zi, center, a1, b1) - x

@jit
def gaussVal1D(alpha, m, center, x, A):
    X = jnp.repeat(jnp.reshape(x, (x.shape[0],1)), alpha.shape[0], axis=1)
    val =  (center-X+0)**m*jnp.exp(-alpha*(center-X)**2)
    #val += (center-X+0)**m*jnp.exp(-alpha*(center-X)**2)
    #val += (center-X+0)**m*jnp.exp(-alpha*(center-X)**2)

    val += (center-X+A)**m*jnp.exp(-alpha*(center-X+A)**2)
    #val += (center-X+A)**m*jnp.exp(-alpha*(center-X+A)**2)
    #val += (center-X+A)**m*jnp.exp(-alpha*(center-X+A)**2) 

    val += (center-X-A)**m*jnp.exp(-alpha*(center-X-A)**2)
    #val += (center-X-A)**m*jnp.exp(-alpha*(center-X-A)**2)
    #val += (center-X-A)**m*jnp.exp(-alpha*(center-X-A)**2)
    return val

@jit
def getError(params, alpha, m, Zi, center, x, a1, b1, Sref):
    L1 = b1-a1

    ##get deformed grid
    S, _ = invMap(params, x, Zi, center, a1, b1)
    Jac, _ = rho_Gauss(params, S, Zi*jnp.ones((1,)), center*jnp.ones((1,)), a1, b1) 
    Jac = (Jac * L1)


    aovals = gaussVal1D(alpha, m, center, S, L1)
    S = jnp.einsum('ai,a,aj->ij', aovals, 1./Jac, aovals) * (x[1]-x[0])

    return jnp.linalg.norm(S-Sref)

def fitDeformation(mf, basMesh):
    A = mf.cell.lattice_vectors()
    a1,b1 = 0.,A[0,0]
    L1 = b1-a1

    denseGrid = 1000

    l = mf.cell._bas[:,1]
    nao = np.sum(l+1)
    alpha, m = np.zeros((nao,)), np.zeros((nao,))

    I = 0
    for i in range(l.shape[0]):
        L = l[i]
        alpha[I:I+L+1] = mf.cell._env[mf.cell._bas[i,5]]
        m[I:I+L+1] = np.arange(0, L+1, 1)
        I += L+1
    center = mf.cell._env[mf.cell._atm[0,1]]
    Zi = mf.cell._atm[0,0]

    alpha, m = jnp.asarray(alpha), jnp.asarray(m)
    denseX = jnp.arange(a1, b1, L1/denseGrid)
    vals = gaussVal1D(alpha, m, center, denseX, L1)

    Sref = vals.T.dot(vals) * L1/denseGrid

    #params = jnp.asarray([0.1, 2., 0.05])
    params = jnp.asarray([2.3099184,   6.475843,   12.867022, 0.05, 1.6527886,  26.1984,    180.28716,0.] )

    x = jnp.arange(a1, b1, L1/basMesh)
    #Jac, normalizedParams = rho_Gauss(params, x, Zi*jnp.ones((1,)), center*jnp.ones((1,)), a1, b1)
    T = Tmap(params,  x, Zi, center, a1, b1)
    S, iter = invMap(params, T, Zi, center, a1, b1)

    print("inversion error ", iter, abs(S-x).max())
    import pdb
    pdb.set_trace()



    x = jnp.arange(a1, b1, L1/basMesh)
    error =  getError(params, alpha, m, Zi, center, x, a1, b1, Sref)

    solver = jaxopt.ProjectedGradient(fun=getError, projection=jaxopt.projection.projection_non_negative)
    #solver = jaxopt.GradientDescent(fun = getError, maxiter = 10, verbose=True)
    res = solver.run(params, alpha, m, Zi, center, x, a1, b1, Sref)

    error =  getError(res.params, alpha, m, Zi, center, x, a1, b1, Sref)

    print(error)
    import pdb
    pdb.set_trace()

    Jac, normalizedParams = rho_Gauss(params, denseX, Zi*jnp.ones((1,)), center*jnp.ones((1,)), a1, b1)
    T = Tmap(params,  denseX, Zi, center, a1, b1)
    S, iter = invMap(params, denseX, Zi, center, a1, b1)

    print("inversion error ", iter)
    plt.plot(denseX, Jac)
    plt.plot(T, 3*jnp.ones((T.shape[0],)), 'o')
    plt.plot(S, 4*jnp.ones((T.shape[0],)), 'o')
    plt.show()

    import pdb
    pdb.set_trace()
    print(jnp.sum(Jac)*L1/denseGrid)

    #getError(params, alpha, m, Zi, center, a1, b1, Sref, basMesh[0])

if __name__ == "__main__":
    '''
    print(np.exp(poch(0.5,2)), hyp22(0.2), jsp.special.gammaln(3.))
    x = -50
    sgn = np.sign(x)
    for n in range(20):
        print( sgn**n * jnp.exp(2.*poch(0.5,n) - 2.*poch(1.5,n) + n * jnp.log(abs(x)) - jsp.special.gammaln(n+1)) )

    print(hyp22((-10)))
    exit(0)
    '''
    nG = 10
    cell = pgto.M(
        a = 8*np.eye(3),
        atom = '''Be 4. 4. 4.''',
        basis = "unc-cc-pvtz",
        verbose = 3,
        mesh = [nG, nG, nG],
        spin=0,
        charge=0,
        unit='B'
    )          

    mf = pyscf.pbc.scf.RHF(cell, exxdiv=None).density_fit(auxbasis='weigend')
    fitDeformation(mf, nG)

