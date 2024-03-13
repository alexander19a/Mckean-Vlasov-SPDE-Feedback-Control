'''
Created on 05.12.2022

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
from scipy import integrate
from CoefficientsAndCost import *
from scipy import fftpack
from GalerkinSetting import laplace

##########################################################################################################################
##########################################Solve Riccati Equation##########################################################
##########################################################################################################################

def Ricc():
    
    V=np.zeros([nt,K,K])
    
    V[nt-1,:,:]=M1
    
    for r in range(nt-2,-1,-1):
        
        print(r)
        
        V[r,:,:]=scipy.linalg.solve(np.identity(K)*(1/float(dt))+np.multiply(V[r+1,:,:],M)+np.transpose(laplace()),V[r+1,:,:]*(1/float(dt))+np.multiply(V[r+1,:,:],-laplace())+Q1)
        #V[r,:,:]=scipy.linalg.solve(np.identity(K)*(1/float(dt))+np.transpose(laplace()),V[r+1,:,:]*(1/float(dt))+np.multiply(V[r+1,:,:],-laplace())-np.multiply(V[r+1,:,:],V[r+1,:,:])+Q1)

    return V

def Ricc2(V):
    
    phi=np.zeros([nt,K])

    for r in range(nt-2,-1,-1):
        print(r)
        
        phi[r,:]=scipy.linalg.solve(np.identity(K)*(1/float(dt))-np.transpose(-laplace()-np.multiply(M,V[r+1,:,:])),B2.dot(phi[r+1,:])+phi[r+1,:]*(1/float(dt)))

    return phi

def Ricc3(V):
    
    V2=np.zeros([nt,K,K])
    V2[nt-1,:,:]=M2
    
    for r in range(nt-2,-1,-1):
                
        V2[r,:,:]=scipy.linalg.solve(np.identity(K)*(1/float(dt))+np.multiply(V2[r+1,:,:],M)-np.transpose(-laplace()+B2-np.multiply(M,V[r+1,:,:])),V2[r+1,:,:]*(1/float(dt))+np.multiply(V2[r+1,:,:],-laplace()+B2-np.multiply(M,V[r+1,:,:]))+(np.multiply(V2[r+1,:,:],B2)+np.multiply(np.transpose(B2),V[r+1,:,:]))+Q2)
        
    return V2

def RiccRef():
    
    V=np.zeros([nt,K,K])
    
    V[nt-1,:,:]=np.identity(K)
    
    for r in range(nt-2,-1,-1):
        
       
        V[r,:,:]=scipy.linalg.solve(np.identity(K)*(1/float(dt))+V[r+1,:,:]+np.transpose(laplace()),V[r+1,:,:]*(1/float(dt))+np.multiply(V[r+1,:,:],-laplace())+np.identity(K))

    return V

def RiccRef2(V,yref):
    
    phi=np.zeros([nt,K])

    for r in range(nt-2,-1,-1):
        print(r)
        
        phi[r,:]=scipy.linalg.solve(np.identity(K)*(1/float(dt))-np.transpose(-laplace()-V[r+1,:,:]),phi[r+1,:]*(1/float(dt))-yref[r+1,:])

    return phi



#solve state equation controlled by Ricc control
def solveStateRiccRef(u0,w,V,V2,phi,yref,yT):

   
    A=sparse.csr_matrix(laplace()+(1/float(dt))*np.identity(K))
    
    out=[]
    
    #initialize
    u=[]
    v=[]
    
    for i in range(Ntest):
        ui=np.zeros([nt,K])
        vi=np.zeros([nt,K])
        ui[0,:]=u0
        vi[0,:]=-np.multiply(np.multiply(Rinv,np.transpose(D)),V[0,:,:]).dot(u0)-np.multiply(np.multiply(Rinv,np.transpose(D)),V2[0,:,:]).dot(u0)-np.multiply(Rinv,np.transpose(D)).dot(phi[0,:])
        
        u.append(ui)
        v.append(vi)
    
    m=u0
    
    for r in range(1,nt):
        print(r)
        m2=np.zeros([K])
        
        for i in range(Ntest):
            
            ypre=u[i][r-1,:]
            
            Vr=sparse.csr_matrix(np.multiply(np.multiply(Rinv,np.transpose(D)),V[r-1,:,:]))
            
            y=spsolve(A,(1/float(dt))*ypre+(1/float(dt))*w[i][r,:]-Vr.dot(ypre)-np.multiply(np.multiply(Rinv,np.transpose(D)),V2[r-1,:,:]).dot(m)-np.multiply(Rinv,np.transpose(D)).dot(phi[r-1,:])+B2.dot(m))
            v[i][r,:]=-np.multiply(np.multiply(Rinv,np.transpose(D)),V[r,:,:]).dot(y)-np.multiply(np.multiply(Rinv,np.transpose(D)),V2[r,:,:]).dot(m)-np.multiply(Rinv,np.transpose(D)).dot(phi[r,:])
            
            u[i][r,:]=y
            
            m2=m2+y
        
        m=m2/float(Ntest)

    out.append(u)
    out.append(v)
    
    return out




def solveState0(u0,w):

    
    A=sparse.csr_matrix(laplace()+(1/float(dt))*np.identity(K))
    
    #initialize solution of particle system
    u=[]
    
    #mean
    m=0
    
    for i in range(Ntest):
        
        ui=np.zeros([nt,K])
    
        ui[0,:]=u0
        
        m=m+u0
        
        u.append(ui)
        
    m=m/float(Ntest)

    #plt.plot(np.sqrt(K/float(L))*tf.signal.idct(m,type=2,norm='ortho'))
    #plt.show()

    for r in range(1,nt):
        
        print(r)
        
        m2=0

        for i in range(N):
            
            ypre=u[i][r-1,:]
            
            y=spsolve(A,(1/float(dt))*ypre+(1/float(dt))*w[i][r,:]+B2.dot(m))
            
            m2=m2+y
            
            u[i][r,:]=y
        
        m=m2/float(Ntest)
        #plt.plot(np.sqrt(K/float(L))*tf.signal.idct(m,type=2,norm='ortho'))
        #plt.show()
    return u



