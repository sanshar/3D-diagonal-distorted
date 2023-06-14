# Copyright (c) 2012  Gerald Knizia
# 
# This file is part of the IR/WMME program
# (See https://sites.psu.edu/knizia/)
# 
# IR/WMME is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
# 
# IR/WMME is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with bfint (LICENSE). If not, see http://www.gnu.org/licenses/

"""A simple implementation of the DIIS (``direct inversion in the iterative subspace'')
convergence accelerator:
   P.Pulay; Chem. Phys. Lett. 73 393 (1980)
   P.Pulay; J. Comput. Chem. 3 556 (1982)
Note: this technique is ubiquitous in quantum chemistry, but has not found
   that much attention elsewhere. A variant of this restricted to linear
   systems was later re-invented by mathematicans and "GMRes" (generalized
   minimum residual) and is more widely known.
--
C. Gerald Knizia, 2012
"""
from __future__ import print_function
from numpy import *
from numpy.linalg import *

class FDiisContext:
    def __init__(self, nDim):
        self.MaxDim = nDim
        self.nDim = 0
        self.iNext = 0
        self.DbgPrint = False
        self.NotApplied = True
        self.iVectorAge = zeros(self.MaxDim,dtype=int)
    def Reset(self):
        self.nDim = 0
        self.iNext = 0
    def __str__(self):
        if self.NotApplied:
            return " -  -"
        else:
            return "%2i %2i" % (self.nDim, self.iNext)

    def RemoveBadVectors(self,iThis):
        # kick out vectors which are too bad to be useful for extrapolation.
        nDim = self.nDim
        Errs1 = self.Errs[:,:nDim]
        B0 = dot(conj(Errs1.T),Errs1)
        EMin = min(diag(B0))

        iVecs = []
        for i in range(nDim):
            if ( B0[i,i].real <= 1e12 * EMin or i == iThis ):
                iVecs.append(i)
        if len(iVecs) != nDim:
            iVecs = array(iVecs)
            nDim = len(iVecs)
            iThis = list(iVecs).index(iThis)
            self.Amps[:,:nDim] = self.Amps[:,iVecs]
            self.Errs[:,:nDim] = self.Errs[:,iVecs]
            if (self.Othr is not None):
                self.Othr[:,:nDim] = self.Othr[:,iVecs]
            self.iVectorAge[:nDim] = self.iVectorAge[iVecs]
            self.nDim = nDim
            iVecs = range(nDim)

    def Apply(self, T_, R_, O_ = None, Skip=None):
        T = T_.flatten()
        R = R_.flatten()

        ContinueIfStarted = True
        if dot(conj(R),R) < 1e-30: # <- otherwise divide by zero in the B scaling.
           Skip = True; ContinueIfStarted = False
        if (Skip is not None and Skip and (self.nDim == 0 or not ContinueIfStarted)):
           # about the last one: this means 'continue with diis if you
           # ever began with it'. might be required if iterations happen to
           # go in the wrong direction.
           self.NotApplied = True
           if (O_ is not None): return T_, R_, O_, 1.0
           else:               return T_, R_, 1.0
        self.NotApplied = False

        def PrintVec(s,T):
           print("!x DIIS: %-10s = %s ..." % (s, " ".join(["%12.6f" % o for o in T[:10]])))
        if self.DbgPrint:
           PrintVec("Input T", T)
           PrintVec("Input R", R)
           print("History:")
           for i in range(self.nDim):
              PrintVec("Amps[%i]" % i, self.Amps[:,i])
           for i in range(self.nDim):
              PrintVec("Errs[%i]" % i, self.Errs[:,i])

        O = None
        if ( O_ is not None ):
            O = O_.flatten()

        if self.nDim == 0:
            self.Amps = zeros((len(T), self.MaxDim),T.dtype)
            self.Errs = zeros((len(R), self.MaxDim),R.dtype)
            if (O is not None):
                self.Othr = zeros((len(O), self.MaxDim),O.dtype)
            else:
                self.Othr = None
        if self.nDim < self.MaxDim:
            self.nDim += 1
        iThis = self.iNext
        for i in range(self.nDim):
            self.iVectorAge[i] += 1
        self.iVectorAge[iThis] = 0


        self.Amps[:,iThis] = T
        self.Errs[:,iThis] = R
        if O is not None:
            self.Othr[:,iThis] = O

        self.RemoveBadVectors(iThis)
        nDim = self.nDim


        Errs1 = self.Errs[:,:nDim]
        B0 = dot(conj(Errs1.T),Errs1)
        if self.DbgPrint:
            print("\n -- DIIS SYSTEM:")
            print("B0 = \n%s" % B0)

        B = zeros((nDim+1,nDim+1),B0.dtype)
        B[:nDim,:nDim] = B0
        rhs = zeros((nDim+1))
        fScale = 0
        for i in range(nDim):
            fScale += log(B[i,i].real)
            B[nDim,i] = -1
            B[i,nDim] = -1
        fScale = exp(fScale/nDim)
        B[:nDim,:nDim] /= fScale
        rhs[nDim] = -1.
        B[nDim,nDim] = 0.
        if False:
            print("\n -- DIIS SYSTEM:")
            print("B = \n%s" % B)
            print("RHS = \n%s" % rhs)
            print("fScale = %8.2e" % fScale)

        if 1:
            # straight diis
            try:
                c1 = solve(B, rhs)
            except LinAlgError as e:
                # I don't think that this is supposed to happen here.
                print("diis: resorted to lstsq...")
                (c1,fitresid,rank,sigma) = lstsq(B, rhs)
        else:
            ew,ev = eigh(B[:-1,:-1])
            c1 = array(list(ev[:,0]) + [ew[0]])
        c = c1[:-1]
        if self.DbgPrint or False:
            print("B = \n%s" % B)
            print("RHS = \n%s" % rhs)
            print("C = \n%s" % c)
            print("c1[-1] = %8.2e" % c1[-1])
            print("fScale = %8.2e" % fScale)

        c /= sum(c) # might have cut out some weight in overlap truncation.
        #print("c[iThis] = %8.2e" % c[iThis])

        #print("output c: %s" % c)
        Tnew = dot(self.Amps[:,:nDim], c[:,newaxis])[:,0]
        Rnew = dot(self.Errs[:,:nDim], c[:,newaxis])[:,0]
        if O is not None:
            Onew = dot(self.Othr[:,:nDim], c[:,newaxis])[:,0]

        if self.nDim < self.MaxDim:
            self.iNext = self.nDim
        else:
            self.iNext = (iThis + 1) % self.nDim


        if self.DbgPrint:
           PrintVec("Output T", Tnew)
           PrintVec("Output R", Rnew)

        if O is not None:
            return Tnew.reshape(T_.shape), Rnew.reshape(R_.shape), Onew.reshape(O_.shape), (abs(c1[-1])*fScale)**.5
        else:
            return Tnew.reshape(T_.shape), Rnew.reshape(R_.shape), (abs(c1[-1])*fScale)**.5

# kate: indent-width 4
