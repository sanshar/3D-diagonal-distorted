from pyscf.pbc import gto as pgto
from pyscf.pbc import df as pdf
from pyscf import gto
import numpy, pyscf, HF, scipy, HF2, HF3, HF4, Distort, Distort2, scipy, HFdirect, temp, Distort3, HFdirect2
import KnotheTransport, KnotheTransportPeriodic
import numpy as np
from pyscf.pbc import pwscf
from pyscf.pbc.dft import multigrid

ng0 = 22
for nG in range(ng0,80,2):
    cell = pgto.M(
        a = 7*numpy.eye(3),
        atom = '''Li 1.5 4. 4.
                  H  4.5 4. 4.''',
                #H    4.6 4.6 4.6''',#+gh,
        #basis = "cc-pvdz",
        #basis = "gth-cc-qzvp", #{'Li':"gth-cc-dzvp", 'H': "gth-cc-dzvp"},
        basis = "gth-qzv3p", #{'Li':"gth-cc-dzvp", 'H': "gth-cc-dzvp"},
        pseudo = "gth-pade",
        verbose = 4,
        mesh = [nG, nG, nG],
        spin=0,
        charge=0,
        unit='B'
    )          

    for g in range(20,50,2):
        cell.mesh=[g,g,g]
        mf = pwscf.KRHF(cell, exxdiv=None)
        mf.kernel()
    #mf = pyscf.pbc.dft.KRKS(cell, [1,1,1]).rs_density_fit(auxbasis='weigend')
    #mf.xc = 'lda'
    #mf.init_guess = '1e'
    #if (nG == ng0):
    #    mf.kernel()
    #exit(0)
    
    #mf = pwscf.KRHF(cell, cell.make_kpts([1,1,1]))
    #mf.kernel()
    
    #mf = cell.KS()
    #mf.xc = 'lda'
    #import pdb
    #pdb.set_trace()
    #mf.kernel()
    #exit(0)
    cell.mesh=[nG,nG,nG]
    if (nG == ng0):
        mf = pyscf.pbc.scf.KRHF(cell, exxdiv=None).rs_density_fit(auxbasis='weigend')
        #mf = pyscf.pbc.dft.KRKS(cell, [1,1,1]).rs_density_fit(auxbasis='weigend')
        #mf.xc = 'lda'
        mf.kernel()


        #'''
        #C = KnotheTransport.LearnTransport(25, 25, 25, mf, 6)
        #KnotheTransport.Plot2DTransport(C, 25, 25, 0.5)
        #Cx, Cy, Cz = KnotheTransport.fitChebyshev(20, 20, 20, C)
        #Fx, Fy, Fz = KnotheTransport.fitFourier(mf, 20, 20, 20, C)

        C = KnotheTransportPeriodic.LearnTransport(15, 15, 15, mf, 12,0.005 )
        #KnotheTransportPeriodic.Plot2DTransport(C, 25, 25, 0.5)
        #Cx, Cy, Cz = KnotheTransportPeriodic.fitChebyshev(20, 20, 20, C)
        Fx, Fy, Fz = KnotheTransportPeriodic.fitFourier(mf, 15, 15, 15, C)
        #np.save("Cx", Cx)
        #np.save("Cy", Cy)
        #np.save("Cz", Cz)
        np.save("Fx", Fx)
        np.save("Fy", Fy)
        np.save("Fz", Fz)
        #'''
    Cx = np.load("Cx.npy")
    Cy = np.load("Cy.npy")
    Cz = np.load("Cz.npy")

    Fx = np.load("Fx.npy")
    Fy = np.load("Fy.npy")
    Fz = np.load("Fz.npy")

    #'''
    #exit(0)
    #K1 = cell.pbc_intor('cint1e_kin_sph')
    #S1 = cell.pbc_intor('cint1e_ovlp_sph')
    #N1 = mf.with_df.get_nuc()
    #N2 = mf.with_df.get_pp()

    #print(S1.diagonal()[:5])
    #print(K1.diagonal()[:5])
    #print(N2.diagonal()[:5])
    #print(N1.diagonal()[:5])
    
    #mf = pyscf.scf.RHF(cell).density_fit(auxbasis='weigend')
    #mf.init_guess = '1e'
    #mf.kernel()

    print("\n\n", cell.mesh)
    #mf = pwscf.KRKS(cell, cell.make_kpts([1,1,1]))
    #mf = pyscf.pbc.scf.RKS(cell) #.density_fit(auxbasis='weigend')
    #mf.xc = 'LDA'
    #mf.kernel()
    #continue

    mf = pyscf.pbc.scf.KRHF(cell, exxdiv=None).rs_density_fit(auxbasis='weigend')
    mf.kernel()

    #invFlow, JacAllFun = Distort3.returnFunc(cell)
    #HFdirect.HFExact(cell, [nG,nG,nG], [nG, nG, nG], mf, invFlow, JacAllFun, productGrid=False)
    #invFlow, JacAllFun, invFlowSinglePoint, JacAllSinglePoint = Distort3.returnFunc2(cell, Cx, Cy, Cz)
    invFlow, JacAllFun, invFlowSinglePoint, JacAllSinglePoint = Distort3.returnFuncFourier(cell, Fx, Fy, Fz)
    #print("FOURIER")
    #HFdirect.HFCheck(cell, [nG,nG,nG], [nG, nG, nG], mf, invFlow, JacAllFun, productGrid=True)
    HFdirect.HF(cell, [nG,nG,nG], [nG, nG, nG], mf, invFlow, JacAllFun, productGrid=True)
    #continue

    '''
    mf = pyscf.pbc.scf.RHF(cell)
    mf.init_guess = '1e'
    #import pdb
    #pdb.set_trace()
    mf.kernel()
    K1 = cell.pbc_intor('cint1e_kin_sph')
    S1 = cell.pbc_intor('cint1e_ovlp_sph')
    N1 = mf.with_df.get_nuc()
    N2 = mf.with_df.get_pp()

    print(S1[1,1], N2[1,1], K1[1,1])
    [d,v] = scipy.linalg.eigh(K1+N2, S1)
    print(d[:5])

    K = cell.intor('int1e_kin')
    S = cell.intor('int1e_ovlp')
    N = cell.intor('int1e_nuc')
    
    [d,v] = scipy.linalg.eigh(K+N, S)
    print(d[:5])
    '''
    '''
    C = np.fromfile("cheb.bin", dtype=np.float64).reshape(10, 10, 10, -1)
    
    a1,b1,a2,b2,a3,b3 = 0.,cell.a[0,0],0.,cell.a[1,1],0.,cell.a[2,2]
    maxIter, alpha, tol = 200, 0.8, 1.e-6

    def invFlow(g):
        x,y,z,_ = Distort.inv_flow(g[:,0], g[:,1], g[:,2], C, a1, b1, a2, b2, a3, b3, maxIter, alpha, tol)
        return np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))

    def Flow(g):
        return Distort.flow(g[:,0], g[:,1], g[:,2], C, a1, b1, a2, b2, a3, b3)

    Tdel = jacfwd(lambda grid : Distort2.flow(grid,C, a1,b1,a2,b2,a3,b3))
    def Jac(grid):
        return vmap(Tdel, in_axes=(0))(grid+1.e-6)
    '''
    #Jac = lambda grid : vmap(Tdel, in_axes=(0))(grid+1e-5)

    #KE =  HF.getKinExact(cell, [nG,nG,nG], [7*nG, 7*nG, 7*nG], invFlow, Flow, Jac)

    #print(KE)
    #mf.kernel()

exit(0)

K = cell.intor('int1e_kin')
S = cell.intor('int1e_ovlp')
N = cell.intor('int1e_nuc')

[d,v] = scipy.linalg.eigh(K+N, S)
print(d[:5])


#mf = pyscf.pbc.scf.KHF(cell, exxdiv=None).rs_density_fit()
mf = pyscf.pbc.scf.KHF(cell, exxdiv="vcut_sph")#.rs_density_fit()


K = cell.pbc_intor('cint1e_kin_sph')
S = cell.pbc_intor('cint1e_ovlp_sph')
N = mf.with_df.get_nuc()

[d,v] = scipy.linalg.eigh(K+N, S)
print(d[:5])

K = cell.intor('cint1e_kin_sph')
S = cell.intor('cint1e_ovlp_sph')
N = cell.intor('cint1e_nuc_sph')

[d,v] = scipy.linalg.eigh(K+N, S)
print(d[:5])
#exit(0)
#mf.kernel()

nB = nG
HF.HF(cell, [nB,nB,nB], [nG, nG, nG], mf)
HF2.HF(cell, [nB,nB,nB], [nG, nG, nG], mf)