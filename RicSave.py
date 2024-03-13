'''
Created on 08.11.2022

@author: alexa
'''


import tensorflow as tf
import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse import spdiags
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from CoefficientsAndCost import *
import pickle
from GalerkinSetting import laplace
from solveRicc import solveStateRiccRef, solveState0
from solveRicc import Ricc, Ricc2, Ricc3
from generateRef import ref
from SolveSpde import solveState
from SolveSpde import solveStateUc


##########################################################################################################################
#################################Calculate Optimal Control for Comparison#################################################
##########################################################################################################################

dtt=1/float(2000)
ntt=int(T/float(dtt))


V=np.zeros([ntt,K,K])
    
V[ntt-1,:,:]=M1
    
for r in range(ntt-2,-1,-1):
        
    print(r)
        
    V[r,:,:]=scipy.linalg.solve(np.identity(K)*(1/float(dtt))+np.multiply(V[r+1,:,:],M)+np.transpose(laplace()),V[r+1,:,:]*(1/float(dtt))+np.multiply(V[r+1,:,:],-laplace())+Q1)
    

dt2=1/float(20)
nt2=int(T/float(dt))

V20=np.zeros([nt2,K,K])

for r in range(nt2):
    V20[r,:,:]=V[r*100,:,:]
    
with open('Ric.pickle', 'wb') as gc:
    pickle.dump(V20,gc, protocol=pickle.HIGHEST_PROTOCOL)

