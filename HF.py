import pyscf, LindseyWavelets, Distort
import numpy as np
from diis import FDiisContext
import matplotlib.pyplot as plt

def getOrbs(basgrid, grid):
    print("hello")

def HF(cell, basMesh, nmesh):
    a1,b1,a2,b2,a3,b3 = 0,cell.a[0,0],0,cell.a[1,1],0,cell.a[2,2]
    grid = cell.get_uniform_grids(nmesh)
    basgrid = cell.get_uniform_grids(basMesh)

    C = np.fromfile("cheb.bin", dtype=np.float64).reshape(20, 20, 20, -1)
    T1, T2, T3, J = Distort.flow(grid[:,0], grid[:,1], grid[:,2], C, a1,b1,a2,b2,a3,b3)


    np1 = 20
    np2 = 20

    # z coordinate
    z = 0.4

    # define uniform plotting grid
    xp1 = np.linspace(a1,b1,np1)
    xp2 = np.linspace(a2,b2,np2)
    X1 = np.repeat(np.reshape(xp1,(np1,1)),np2,axis=1)
    X2 = np.repeat(np.reshape(xp2,(1,np2)),np1,axis=0)
    Xp1 = np.reshape(X1,np1*np2)
    Xp2 = np.reshape(X2,np1*np2)
    Xp3 = z*np.ones(np1*np2)

    #T1, T2, T3, J = Distort.flow(Xp1, Xp2, Xp3, C, a1,b1,a2,b2,a3,b3)

    fig, ax = plt.subplots()
    ax.scatter(T2[7*225:8*225],T3[7*225:8*225])
    #ax.scatter(T1,T2)
    ax.set_box_aspect(1)
    plt.show()

    import pdb
    pdb.set_trace()
    orbs = getOrbs(basMesh, grid)

