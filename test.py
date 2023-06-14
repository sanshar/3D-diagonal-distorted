from pyscf.pbc import gto as pgto
from pyscf.pbc import df as pdf
from pyscf import gto
import numpy, pyscf, HF

nk = 10
cell = pgto.M(
    a = 6*numpy.eye(3),
    atom = '''H 3.4 3.4 3.4,
              H    2.6 2.6 2.6''',#+gh,
    basis = "gth-dzv",
    verbose = 5,
    mesh = [2*nk+1, 2*nk+1, 2*nk+1],
#    precision=1.e-16
)          

ks = pyscf.pbc.scf.KHF(cell)
ks.kernel()

HF.HF(cell, [5,5,5], [15,15,15])