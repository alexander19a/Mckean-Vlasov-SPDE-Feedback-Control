'''
Created on 03.11.2022

@author: alexa
'''
import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse import spdiags
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from GalerkinSetting import laplace
import tensorflow as tf
from CoefficientsAndCost import *

##########################################################################################################################
#################################Generates a Reference Profile############################################################
##########################################################################################################################

#laplacian for given boundary conditions
A=sparse.csr_matrix(laplace()+(1/float(dt))*np.identity(K))

def ref(u0):
    
    
    #initialize
    u=np.zeros([nt,K])

    u[0,:]=u0
    
    ypre=u0
    

    for r in range(1,nt):
        

   
        #compute non-linearity for current time step
        nonlin=f(np.sqrt(K/float(L))*scipy.fftpack.idct(ypre,type=2, norm='ortho'))
    
        
        fFourier=np.sqrt(L/float(K))*scipy.fftpack.dct(nonlin,type=2, norm='ortho')
       
        
        y=spsolve(A,(1/float(dt))*ypre+fFourier)
        
        
        u[r,:]=y
        ypre=y
        
    return u



