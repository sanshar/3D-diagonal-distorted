from pyscf.pbc import gto as pgto
from pyscf.pbc import df as pdf
from pyscf import gto
import numpy, pyscf, scipy, scipy, HFdirect
import KnotheTransportPeriodic_finufft, KnotheTransportPeriodic
import numpy as np
from pyscf.pbc.dft import multigrid


ng0 = 10
for nG in range(ng0,200,4):
    cell = pgto.M(
        a = 10*numpy.eye(3),
        atom = '''Be 5. 5. 5.''',
        basis = "cc-pvtz",
        #pseudo = "gth-pade",
        verbose = 3,
        mesh = [nG, nG, nG],
        spin=0,
        charge=0,
        unit='B'
    )          

    cell.mesh=[nG,nG,nG]

    if (nG == ng0):

        mf = pyscf.pbc.scf.RHF(cell, exxdiv=None).mix_density_fit()#auxbasis='weigend')
        mf.kernel()

        C = KnotheTransportPeriodic_finufft.LearnTransportInverse(20, 20, 20, mf, 15, 0.0)
        np.save("fullC", C)
        #C = np.load("fullC.npy").reshape(20,20,20,-1)
        #print("Solving done")
        #KnotheTransportPeriodic.Plot2DTransportInverseFit(C, 25, 25, 5., mf)
        


    print("\n\n", cell.mesh)

    a1,b1,a2,b2,a3,b3 = 0.,mf.cell.a[0,0],0.,mf.cell.a[1,1],0.,mf.cell.a[2,2]
    flow = lambda x, y, z : KnotheTransportPeriodic_finufft.inv_flow(x, y, z, C, a1, b1, a2, b2, a3, b3, C.shape[-1])
    invFlow = lambda x, y, z : KnotheTransportPeriodic_finufft.flowWithJac(x, y, z, C, a1, b1, a2, b2, a3, b3, C.shape[-1])
    Jac = lambda x, y, z, : KnotheTransportPeriodic_finufft.getJac(x, y, z, C, a1, b1, a2, b2, a3, b3, C.shape[-1])

    HFdirect.HF(cell, [nG,nG,nG], [nG, nG, nG], mf, invFlow, flow, Jac, allElectronPS = 4)
    print(cell.basis)
    #HFdirect.HF_ISDF_DistrotedGrid(cell, [nG,nG,nG], [nG, nG, nG], mf, invFlow, flow, Jac)
