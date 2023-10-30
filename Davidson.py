from functools import reduce
import numpy
import scipy.linalg

SAFE_EIGH_LINDEP = 1e-15
DAVIDSON_LINDEP = 1e-14
DSOLVE_LINDEP = 1e-15
MAX_MEMORY = 200000

# sort by similarity has problem which flips the ordering of eigenvalues when
# the initial guess is closed to excited state.  In this situation, function
# _sort_by_similarity may mark the excited state as the first eigenvalue and
# freeze the first eigenvalue.
SORT_EIG_BY_SIMILARITY =False
# Projecting out converged eigenvectors has problems when conv_tol is loose.
# In this situation, the converged eigenvectors may be updated in the
# following iterations.  Projecting out the converged eigenvectors may lead to
# large errors to the yet converged eigenvectors.
PROJECT_OUT_CONV_EIGS = False

FOLLOW_STATE = False


def _qr(xs, dot, lindep=1e-14):
    '''QR decomposition for a list of vectors (for linearly independent vectors only).
    xs = (r.T).dot(qs)
    '''
    nvec = len(xs)
    dtype = xs[0].dtype
    qs = numpy.empty((nvec,xs[0].size), dtype=dtype)
    rmat = numpy.empty((nvec,nvec), order='F', dtype=dtype)

    nv = 0
    for i in range(nvec):
        xi = numpy.array(xs[i], copy=True)
        rmat[:,nv] = 0
        rmat[nv,nv] = 1
        for j in range(nv):
            prod = dot(qs[j].conj(), xi)
            xi -= qs[j] * prod
            rmat[:,nv] -= rmat[:,j] * prod
        innerprod = dot(xi.conj(), xi).real
        norm = numpy.sqrt(innerprod)
        if innerprod > lindep:
            qs[nv] = xi/norm
            rmat[:nv+1,nv] /= norm
            nv += 1
    return qs[:nv], numpy.linalg.inv(rmat[:nv,:nv])

def _fill_heff_hermitian(heff, xs, ax, xt, axt, dot):
    nrow = len(axt)
    row1 = len(ax)
    row0 = row1 - nrow
    for ip, i in enumerate(range(row0, row1)):
        for jp, j in enumerate(range(row0, i)):
            heff[i,j] = dot(xt[ip].conj(), axt[jp])
            heff[j,i] = heff[i,j].conj()
        heff[i,i] = dot(xt[ip].conj(), axt[ip]).real

    for i in range(row0):
        axi = numpy.asarray(ax[i])
        for jp, j in enumerate(range(row0, row1)):
            heff[j,i] = dot(xt[jp].conj(), axi)
            heff[i,j] = heff[j,i].conj()
        axi = None
    return heff

def _gen_x0(v, xs):
    space, nroots = v.shape
    x0 = numpy.einsum('c,x->cx', v[space-1], numpy.asarray(xs[space-1]))
    for i in reversed(range(space-1)):
        xsi = numpy.asarray(xs[i])
        for k in range(nroots):
            x0[k] += v[i,k] * xsi
    return x0

def _sort_elast(elast, conv_last, vlast, v, fresh_start):
    '''
    Eigenstates may be flipped during the Davidson iterations.  Reorder the
    eigenvalues of last iteration to make them comparable to the eigenvalues
    of the current iterations.
    '''
    if fresh_start:
        return elast, conv_last
    head, nroots = vlast.shape
    ovlp = abs(numpy.dot(v[:head].conj().T, vlast))
    idx = numpy.argmax(ovlp, axis=1)

    return [elast[i] for i in idx], [conv_last[i] for i in idx]

def Davidson(actH, precondition, inArray, subspace = 6, nroots = 1, niter=100, thresh=1.e-4):
    Thresh = max(1.e-8, thresh)
    subspace = nroots + subspace

    if (len(inArray.shape) > 1 and inArray.shape[0] != nroots):
        if (len(inArray.shape) != 2):
            print ("inut array should be a 2-d array, instead it is {0:d}-d array".format(len(inArray.shape)))
            exit(0)
        print ("Only {0:d} input vectors provided which is fewer than the number of roots needed :{1:d}".format(inArray.shape[0], nroots))
        exit(0)
    nbas = inArray.shape[1]

    hsub = numpy.zeros((subspace, subspace), dtype=complex)
    v = numpy.zeros((subspace, nbas), dtype=complex, order='C')
    sigma = numpy.zeros((subspace, nbas), dtype=complex, order='C')


    q, r = numpy.linalg.qr(inArray.T)
    v[:nroots] = 1.*q.T

    for i in range(nroots):
        sigma[i] = actH(v[i])
        #actH(v[i], sigma[i], kpt)
        hsub[:nroots,i] = numpy.dot(v[:nroots].conj(), sigma[i])

    e,hv = numpy.linalg.eigh(hsub[:nroots, :nroots])
    EigVal = e[:nroots]
    sigma[:nroots] = numpy.einsum('ij,ik->jk',hv[:nroots,:nroots], sigma[:nroots])
    v[:nroots] = numpy.einsum('ij,ik->jk',hv[:nroots,:nroots], v[:nroots])

    hsub = 0.*hsub
    for i in range(nroots):
        hsub[i,i] = EigVal[i]

    iter, root, state = 0, 0, nroots
    r = sigma[:nroots] - numpy.einsum('i,ij->ij', EigVal, v[:nroots])
    error = numpy.zeros((nroots,))
    #Thresh = min(1.e-1, numpy.linalg.norm(r)/3.)
    while (iter < niter*nroots):

        maxerror = 0.
        for i in range(nroots):
            error[i] = numpy.linalg.norm(r[i])
            maxerror = max(maxerror, error[i])
            if (error[i] > Thresh):
                root = i
                break
        if (maxerror < Thresh):
            break

        if (precondition is not None):            
            r[root] = precondition(r[root], e[root])

        ovlp = numpy.dot(v[:state].conj(), r[root])
        r[root] = r[root] - numpy.dot(ovlp, v[:state])
        v[state] = r[root]/numpy.linalg.norm(r[root])

        sigma[state] = actH(v[state])
        for i in range(state+1):
            hsub[i,state] = numpy.dot(v[i].conj(), sigma[state])
            hsub[state, i] = hsub[i,state].conj()

        e,hv = numpy.linalg.eigh(hsub[:state+1, :state+1])
        if (state == subspace-1) :  
            sigma[:nroots] = numpy.einsum('ij,ik->jk',hv[:,:nroots], sigma)
            v[:nroots] = numpy.einsum('ij,ik->jk',hv[:,:nroots], v)
            hsub *= 0.
            for i in range(nroots+1):
                hsub[i,i] = e[i]
            state = nroots 
        else:
            state +=1
            sigma[:state] = numpy.einsum('ij,ik->jk',hv, sigma[:state])
            v[:state] = numpy.einsum('ij,ik->jk',hv, v[:state])
            hsub *= 0.
            for i in range(state):
                hsub[i,i] = e[i]

        EigVal = e[:nroots]

        r = sigma[:nroots] - numpy.einsum('i,ij->ij', EigVal, v[:nroots])

        print(EigVal, np.linalg.norm(r))

        iter+=1

    #for i in range(root+1):
    #    print("{0:8d}  {1:18.9f}  {2:18.9f}".format(iter, EigVal[i], error[i]))
    #exit(0)
    return v[:nroots], EigVal, iter, maxerror

def davidson1(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
              lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
              dot=numpy.dot, callback=None,
              nroots=1, lessio=False, pick=None, verbose=0,
              follow_state=FOLLOW_STATE, tol_residual=None,
              fill_heff=_fill_heff_hermitian):
    r'''Davidson diagonalization method to solve  a c = e c.  Ref
    [1] E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
    [2] http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter11.pdf

    Note: This function has an overhead of memory usage ~4*x0.size*nroots

    Args:
        aop : function([x]) => [array_like_x]
            Matrix vector multiplication :math:`y_{ki} = \sum_{j}a_{ij}*x_{jk}`.
        x0 : 1D array or a list of 1D arrays or a function to generate x0 array(s)
            Initial guess.  The initial guess vector(s) are just used as the
            initial subspace bases.  If the subspace is smaller than "nroots",
            eg 10 roots and one initial guess, all eigenvectors are chosen as
            the eigenvectors during the iterations.  The first iteration has
            one eigenvector, the next iteration has two, the third iterastion
            has 4, ..., until the subspace size > nroots.
        precond : diagonal elements of the matrix or  function(dx, e, x0) => array_like_dx
            Preconditioner to generate new trial vector.
            The argument dx is a residual vector ``a*x0-e*x0``; e is the current
            eigenvalue; x0 is the current eigenvector.

    Kwargs:
        tol : float
            Convergence tolerance.
        max_cycle : int
            max number of iterations.
        max_space : int
            space size to hold trial vectors.
        lindep : float
            Linear dependency threshold.  The function is terminated when the
            smallest eigenvalue of the metric of the trial vectors is lower
            than this threshold.
        max_memory : int or float
            Allowed memory in MB.
        dot : function(x, y) => scalar
            Inner product
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.
        nroots : int
            Number of eigenvalues to be computed.  When nroots > 1, it affects
            the shape of the return value
        lessio : bool
            How to compute a*x0 for current eigenvector x0.  There are two
            ways to compute a*x0.  One is to assemble the existed a*x.  The
            other is to call aop(x0).  The default is the first method which
            needs more IO and less computational cost.  When IO is slow, the
            second method can be considered.
        pick : function(w,v,nroots) => (e[idx], w[:,idx], idx)
            Function to filter eigenvalues and eigenvectors.
        follow_state : bool
            If the solution dramatically changes in two iterations, clean the
            subspace and restart the iteration with the old solution.  It can
            help to improve numerical stability.  Default is False.

    Returns:
        conv : bool
            Converged or not
        e : list of floats
            The lowest :attr:`nroots` eigenvalues.
        c : list of 1D arrays
            The lowest :attr:`nroots` eigenvectors.

    Examples:

    >>> from pyscf import lib
    >>> a = numpy.random.random((10,10))
    >>> a = a + a.T
    >>> aop = lambda xs: [numpy.dot(a,x) for x in xs]
    >>> precond = lambda dx, e, x0: dx/(a.diagonal()-e)
    >>> x0 = a[0]
    >>> e, c = lib.davidson(aop, x0, precond, nroots=2)
    >>> len(e)
    2
    '''

    if tol_residual is None:
        toloose = numpy.sqrt(tol)
    else:
        toloose = tol_residual
    #log.debug1('tol %g  toloose %g', tol, toloose)

    if callable(x0):  # lazy initialization to reduce memory footprint
        x0 = x0()
    if isinstance(x0, numpy.ndarray) and x0.ndim == 1:
        x0 = [x0]
    #max_cycle = min(max_cycle, x0[0].size)
    max_space = max_space + (nroots-1) * 3
    # max_space*2 for holding ax and xs, nroots*2 for holding axt and xt
    _incore = max_memory*1e6/x0[0].nbytes > max_space*2+nroots*3
    lessio = lessio and not _incore
    #log.debug1('max_cycle %d  max_space %d  max_memory %d  incore %s',
    #           max_cycle, max_space, max_memory, _incore)
    dtype = None
    heff = None
    fresh_start = True
    e = 0
    v = None
    conv = [False] * nroots
    emin = None
    norm_min = 1

    for icyc in range(max_cycle):
        if fresh_start:
            if _incore:
                xs = []
                ax = []
            else:
                xs = _Xlist()
                ax = _Xlist()
            space = 0
# Orthogonalize xt space because the basis of subspace xs must be orthogonal
# but the eigenvectors x0 might not be strictly orthogonal
            xt = None
            x0len = len(x0)
            xt, x0 = _qr(x0, dot, lindep)[0], None
            #if len(xt) != x0len:
            #log.warn('QR decomposition removed %d vectors.  The davidson may fail.',
            #         x0len - len(xt))
            #if callable(pick):
            #    log.warn('Check to see if `pick` function %s is providing '
            #             'linear dependent vectors', pick.__name__)
            max_dx_last = 1e9
            if SORT_EIG_BY_SIMILARITY:
                conv = [False] * nroots
        elif len(xt) > 1:
            xt = _qr(xt, dot, lindep)[0]
            xt = xt[:40]  # 40 trial vectors at most

        axt = aop(xt)
        for k, xi in enumerate(xt):
            xs.append(xt[k])
            ax.append(axt[k])
        rnow = len(xt)
        head, space = space, space+rnow

        if dtype is None:
            try:
                dtype = numpy.result_type(axt[0], xt[0])
            except IndexError:
                dtype = numpy.result_type(ax[0].dtype, xs[0].dtype)
        if heff is None:  # Lazy initilize heff to determine the dtype
            heff = numpy.empty((max_space+nroots,max_space+nroots), dtype=dtype)
        else:
            heff = numpy.asarray(heff, dtype=dtype)

        elast = e
        vlast = v
        conv_last = conv

        fill_heff(heff, xs, ax, xt, axt, dot)
        xt = axt = None
        w, v = scipy.linalg.eigh(heff[:space,:space])
        if callable(pick):
            w, v, idx = pick(w, v, nroots, locals())
        if SORT_EIG_BY_SIMILARITY:
            e, v = _sort_by_similarity(w, v, nroots, conv, vlast, emin)
            if elast.size != e.size:
                de = e
            else:
                de = e - elast
        else:
            e = w[:nroots]
            v = v[:,:nroots]

        x0 = None
        x0 = _gen_x0(v, xs)
        if lessio:
            ax0 = aop(x0)
        else:
            ax0 = _gen_x0(v, ax)

        if SORT_EIG_BY_SIMILARITY:
            dx_norm = [0] * nroots
            xt = [None] * nroots
            for k, ek in enumerate(e):
                if not conv[k]:
                    xt[k] = ax0[k] - ek * x0[k]
                    dx_norm[k] = numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                    if abs(de[k]) < tol and dx_norm[k] < toloose:
                        #log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                        #          k, dx_norm[k], ek, de[k])
                        conv[k] = True
        else:
            elast, conv_last = _sort_elast(elast, conv_last, vlast, v,
                                           fresh_start)
            de = e - elast
            dx_norm = []
            xt = []
            conv = [False] * nroots
            for k, ek in enumerate(e):
                xt.append(ax0[k] - ek * x0[k])
                dx_norm.append(numpy.sqrt(dot(xt[k].conj(), xt[k]).real))
                conv[k] = abs(de[k]) < tol and dx_norm[k] < toloose
                #if conv[k] and not conv_last[k]:
                #    log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                #              k, dx_norm[k], ek, de[k])
        ax0 = None
        max_dx_norm = max(dx_norm)
        ide = numpy.argmax(abs(de))
        if all(conv):
            #log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
            #          icyc, space, max_dx_norm, e, de[ide])
            if (verbose == 1):
                print('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g'%(icyc, space, max_dx_norm, e, de[ide]))
            break
        elif (follow_state and max_dx_norm > 1 and
              max_dx_norm/max_dx_last > 3 and space > nroots*1):
            #log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
            #          icyc, space, max_dx_norm, e, de[ide], norm_min)
            #log.debug('Large |r| detected, restore to previous x0')
            x0 = _gen_x0(vlast, xs)
            fresh_start = True
            continue

        if SORT_EIG_BY_SIMILARITY:
            if any(conv) and e.dtype == numpy.double:
                emin = min(e)

        # remove subspace linear dependency
        if any(((not conv[k]) and n**2>lindep) for k, n in enumerate(dx_norm)):
            for k, ek in enumerate(e):
                if (not conv[k]) and dx_norm[k]**2 > lindep:
                    xt[k] = precond(xt[k], e[0])
                    xt[k] *= 1/numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                else:
                    xt[k] = None
        else:
            for k, ek in enumerate(e):
                if dx_norm[k]**2 > lindep:
                    xt[k] = precond(xt[k], e[0])
                    xt[k] *= 1/numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                else:
                    xt[k] = None
                    #log.debug1('Throwing out eigenvector %d with norm=%4.3g', k, dx_norm[k])
        xt = [xi for xi in xt if xi is not None]

        for i in range(space):
            xsi = numpy.asarray(xs[i])
            for xi in xt:
                xi -= xsi * dot(xsi.conj(), xi)
            xsi = None
        norm_min = 1
        for i,xi in enumerate(xt):
            norm = numpy.sqrt(dot(xi.conj(), xi).real)
            if norm**2 > lindep:
                xt[i] *= 1/norm
                norm_min = min(norm_min, norm)
            else:
                xt[i] = None
        xt = [xi for xi in xt if xi is not None]
        xi = None
        if verbose :
            print('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g' %(
                    icyc, space, max_dx_norm, e, de[ide], norm_min))
        #log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
        #          icyc, space, max_dx_norm, e, de[ide], norm_min)
        if len(xt) == 0:
            #log.debug('Linear dependency in trial subspace. |r| for each state %s',
            #          dx_norm)
            conv = [conv[k] or (norm < toloose) for k,norm in enumerate(dx_norm)]
            break

        max_dx_last = max_dx_norm
        fresh_start = space+nroots > max_space

        if callable(callback):
            callback(locals())

    x0 = [x for x in x0]  # nparray -> list

    # Check whether the solver finds enough eigenvectors.
    h_dim = x0[0].size
    if len(x0) < min(h_dim, nroots):
        # Two possible reasons:
        # 1. All the initial guess are the eigenvectors. No more trial vectors
        # can be generated.
        # 2. The initial guess sits in the subspace which is smaller than the
        # required number of roots.
        msg = 'Not enough eigenvectors (len(x0)=%d, nroots=%d)' % (len(x0), nroots)
        warnings.warn(msg)

    return numpy.asarray(conv), e, x0
