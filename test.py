from pyscf.pbc import gto as pgto
from pyscf.pbc import df as pdf
from pyscf import gto
import numpy, pyscf, HF, scipy

nG = 60
cell = pgto.M(
    a = 10*numpy.eye(3),
    atom = '''H 5.4 5.4 5.4,
              H    4.6 4.6 4.6''',#+gh,
    #atom = '''H 0.0 0.0 0.0''',
    basis = "cc-pvqz",
    verbose = 3,
    mesh = [nG, nG, nG],
    unit='B'
)          

mf = pyscf.pbc.scf.KHF(cell, exxdiv=None).rs_density_fit()


K = cell.pbc_intor('cint1e_kin_sph')
S = cell.pbc_intor('cint1e_ovlp_sph')
N = mf.with_df.get_nuc()

[d,v] = scipy.linalg.eigh(K+N, S)
print(d[:5])
mf.kernel()

HF.HF(cell, [15,15,15], [nG, nG, nG], mf)