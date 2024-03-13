'''
Created on 03.11.2022

@author: alexa
'''

from CoefficientsAndCost import *

##########################################################################################################################
#############################################Setting for Galerkin Approximation###########################################
##########################################################################################################################

#define laplacian derivative matrix depending on boundary condition
def laplace():
    a=np.identity(K)
    
    #Dirichlet boundary conditions
    if bd==0:
        l=np.arange(1,K+1,1)
        a=np.pi**2*np.multiply(l**2,a/float(L)**2)
    
    #Neumann boundary conditions  
    if bd==1:
        l=np.arange(0,K,1)
        a=np.pi**2*np.multiply(l**2,a/float(L)**2)

  
    
    return a

     