from pyscf.pbc import gto as pgto
from pyscf.pbc import df as pdf
from pyscf import gto
import numpy, pyscf, scipy, HFdirect, Distort3
import KnotheTransport, KnotheTransportPeriodic
import numpy as np
#from pyscf.pbc import pwscf
from pyscf.pbc.dft import multigrid


lat = 4.000 * numpy.eye(3)



nG = 30
cell = pgto.M(
    a = lat,
    atom = '''
Li     0.800000000         0.800000000         0.800000000
Li     0.800000000         2.800087023         2.800087023
Li     2.800087023         0.800000000         2.800087023
Li     2.800087023         2.800087023         0.800000000
H      2.800087023         0.800000000         0.800000000
H      2.800087023         2.800087023         2.800087023
H      0.800000000         0.800000000         2.800087023
H      0.800000000         2.800087023         0.800000000''',
    pseudo = 'gth-pade',                     
    basis = 'gth-CC-dzvp',
    #basis = 'gth-CC-Qzvp',
    unit = 'a',
    verbose =4,
    mesh=[nG,nG,nG]                                    
)

'''
for ng in range(20,70,2):
    cell.mesh=[ng,ng,ng]
    mf = pwscf.KRHF(cell, cell.make_kpts([1,1,1]), exxdiv=None)
    #mf = pyscf.pbc.dft.KRKS(cell, [1,1,1]).rs_density_fit(auxbasis='weigend')
    #mf.xc = 'lda'
    #mf.init_guess = '1e'
    mf.kernel()
'''
mf = pyscf.pbc.scf.RHF(cell, exxdiv=None).rs_density_fit(auxbasis='weigend')
mf.kernel()

ng0 = 16
for nG in range(ng0,200,2):

    cell.mesh = [nG, nG,nG]
    if (nG == ng0):
        #cell.mesh = [30,30,30]
        #mf = pyscf.pbc.dft.KRKS(cell, [1,1,1])#.density_fit(auxbasis='weigend')
        #mf.xc = 'lda'
        #mf.kernel()
        #cell.mesh = [nG, nG,nG]

        mf = pyscf.pbc.scf.RHF(cell, exxdiv=None).rs_density_fit(auxbasis='weigend')
        mf.kernel()
        C = KnotheTransportPeriodic.LearnTransport(10, 10, 10, mf, 12, 0.005)
        #KnotheTransportPeriodic.Plot2DTransport(C, 25, 25, 0.5)
        #Cx, Cy, Cz = KnotheTransport.fitChebyshev(15, 15, 15, C)
        Fx, Fy, Fz = KnotheTransportPeriodic.fitFourier(mf, 20, 20, 20, C)
        #np.save("Cx", Cx)
        #np.save("Cy", Cy)
        #np.save("Cz", Cz)
        np.save("Fx", Fx)
        np.save("Fy", Fy)
        np.save("Fz", Fz)
        #'''
    #Cx = np.load("Cx.npy")
    #Cy = np.load("Cy.npy")
    #Cz = np.load("Cz.npy")

    Fx = np.load("Fx.npy")
    Fy = np.load("Fy.npy")
    Fz = np.load("Fz.npy")


    print(cell.mesh)
    #mf = pwscf.KRHF(cell, cell.make_kpts([1,1,1]))
    #mf = pyscf.pbc.scf.RHF(cell).density_fit(auxbasis='weigend')
    #mf.kernel()

    invFlow, JacAllFun, invFlowSinglePoint, JacAllSinglePoint = Distort3.returnFuncFourier(cell, Fx, Fy, Fz)
    #HFdirect.HF(cell, [nG,nG,nG], [nG, nG, nG], mf, invFlow, JacAllFun, invFlowSinglePoint, JacAllSinglePoint, productGrid=True)
    HFdirect.HF(cell, [nG,nG,nG], [nG, nG, nG], mf, invFlow, JacAllFun, productGrid=True)

    
    
