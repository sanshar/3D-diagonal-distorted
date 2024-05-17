import jaxopt, jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import value_and_grad, vmap, lax, jit, random
import matplotlib.pyplot as plt


@jit
def rho(x):
  return (jsp.special.erf((x+1e-6) *2/ 0.1) - jsp.special.erf((x+1e-6) /2.))/(x+1e-6) 

@jit
def getGaussian(params, xvals):

  @jit
  def fun(params): 
    fval = 0.*xvals
    N = params.shape[0]//2
    for i in range(N):
        fval += params[i] * jnp.exp(-params[i+N]*xvals**2)
         
    return jnp.linalg.norm(rho(xvals) - fval)
  
  #solver = jaxopt.LBFGS(fun=fun, maxiter=20)
  solver = jaxopt.GradientDescent(fun=fun, maxiter=500)
  #solver = jaxopt.NonlinearCG(fun=fun, method="polak-ribiere", maxiter=500)
  res = solver.run(params)  
  return res

N = 4
a = jnp.asarray(np.random.random((2*N,)))
x = jnp.arange(-4,4,8/100)
aout = getGaussian(a, x)
print(aout.params)

w = aout.params[:N]
p = aout.params[N:]
f1 = rho(x)
f2 = 0*f1
for i in range(N):
    f2 += w[i] * jnp.exp(-p[i]*x**2)

plt.plot(x, f1)
plt.plot(x, f2)
plt.show()
