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
######################################################Solve SPDE##########################################################
##########################################################################################################################


#laplacian for given boundary conditions
A=sparse.csr_matrix(laplace()+(1/float(dt))*np.identity(K))

#solve state equation. equation: dx=[Ax+f(x)+B2*E[x]+D*g]dt+eps*BdW
def solveState(u0,model,w,solMin,solMax,normal):
    
    #output
    out=[]
    
    #initialize solution of particle system
    u=[]
    
    #initialize mean of solution
    mu=np.zeros([nt,K])
    
    #mean
    m=np.zeros([K])
    
    for i in range(N):
        
        ui=np.zeros([nt,K])
    
        ui[0,:]=u0
        
        m=m+u0
        
        u.append(ui)
        
    m=m/float(N)
    mu[0,:]=m

    for r in range(1,nt):
        
        
        m2=np.zeros([K])

        for i in range(N):
            
            ypre=u[i][r-1,:]
            
            #compute non-linearity for current time step
            nonlin=f(np.sqrt(K/float(L))*scipy.fftpack.idct(ypre,type=2, norm='ortho'))
            
            #compute Fourier coefficients of non-linearity
            fFourier=np.sqrt(L/float(K))*scipy.fftpack.dct(nonlin,type=2, norm='ortho')

            #evaluate stochastic control at current time step
            r1=tf.cast(ti[r-1], tf.double) 

            x1=tf.Variable([scale2*np.multiply(ypre-solMin,normal)])

            r1=tf.Variable([[scale*r1]])
        
            #evaluate feedback control g_t=G_{model(theta)}(t,x), where model is NN, theta are network parameters
            control=tf.reshape(model(tf.concat([r1,x1], axis=1)),[K])
        
        
            y=spsolve(A,(1/float(dt))*ypre+fFourier+(1/float(dt))*w[i][r,:]+B2.dot(m)+D.dot(control))
        
            
            m2=m2+y
            
            u[i][r,:]=y
        
        m=m2/float(N)
        mu[r,:]=m
        
    out.append(u)
    out.append(mu)
        
    return out

#solve state equation for test case with higher number of samples/particles
def solveStateTest(u0,model,w,solMin,solMax,normal):
    
    #output
    out=[]
    
    #initialize solution of particle system
    u=[]
    
    #initialize mean of solution
    mu=np.zeros([nt,K])
    
    #mean
    m=np.zeros([K])
    
    for i in range(Ntest):
        
        ui=np.zeros([nt,K])
    
        ui[0,:]=u0
        
        m=m+u0
        
        u.append(ui)
        
    m=m/float(Ntest)
    mu[0,:]=m

    for r in range(1,nt):
        
        m2=np.zeros([K])

        for i in range(Ntest):
            
            ypre=u[i][r-1,:]
            
            #compute non-linearity for current time step
            nonlin=f(np.sqrt(K/float(L))*scipy.fftpack.idct(ypre,type=2, norm='ortho'))
            
            #compute Fourier coefficients of non-linearity
            fFourier=np.sqrt(L/float(K))*scipy.fftpack.dct(nonlin,type=2, norm='ortho')
        
            #evaluate stochastic control at current time step
            r1=tf.cast(ti[r-1], tf.double) 

            x1=tf.Variable([scale2*np.multiply(ypre-solMin,normal)])

            r1=tf.Variable([[scale*r1]])
            
            
            #evaluate feedback control g_t=G_{model(theta)}(t,x), where model is NN, theta are network parameters
            control=tf.reshape(model(tf.concat([r1,x1], axis=1)),[K])
        
        
            y=spsolve(A,(1/float(dt))*ypre+fFourier+(1/float(dt))*w[i][r,:]+B2.dot(m)+D.dot(control))
            
            m2=m2+y
            
            u[i][r,:]=y
        
        m=m2/float(Ntest)
        mu[r,:]=m
        
    out.append(u)
    out.append(mu)
        
    return out

#solve uncontrolled state equation
def solveStateUc(u0,w):
    
    
    #initialize solution of particle system
    u=[]
    
    #mean
    m=np.zeros([K])
    
    for i in range(N):
        
        ui=np.zeros([nt,K])
    
        ui[0,:]=u0
        
        m=m+u0
        
        u.append(ui)
        
    m=m/float(N)
    
    for r in range(1,nt):
        
        print(r)
        
        m2=np.zeros([K])

        for i in range(N):
            
            ypre=u[i][r-1,:]
            
            #compute non-linearity for current time step
            nonlin=f(np.sqrt(K/float(L))*scipy.fftpack.idct(ypre,type=2, norm='ortho'))
    
        
            fFourier=np.sqrt(L/float(K))*scipy.fftpack.dct(nonlin,type=2, norm='ortho')
    
            
            y=spsolve(A,(1/float(dt))*ypre+fFourier+(1/float(dt))*w[i][r,:]+B2.dot(m))
            
            m2=m2+y
            
            u[i][r,:]=y
        
        m=m2/float(N)
    
    return u

#solve uncontrolled state equation for test case with higher number of samples/particles
def solveStateUcTest(u0,w):
    
    
    #initialize solution of particle system
    u=[]
    
    #mean
    m=np.zeros([K])
    
    for i in range(Ntest):
        
        ui=np.zeros([nt,K])
    
        ui[0,:]=u0
        
        m=m+u0
        
        u.append(ui)
        
    m=m/float(Ntest)
    
    for r in range(1,nt):
        
        print(r)
        
        m2=np.zeros([K])

        for i in range(Ntest):
            
            ypre=u[i][r-1,:]
            
            #compute non-linearity for current time step
            nonlin=f(np.sqrt(K/float(L))*scipy.fftpack.idct(ypre,type=2, norm='ortho'))
    
        
            fFourier=np.sqrt(L/float(K))*scipy.fftpack.dct(nonlin,type=2, norm='ortho')
    
            
            y=spsolve(A,(1/float(dt))*ypre+fFourier+(1/float(dt))*w[i][r,:]+B2.dot(m))
            
            m2=m2+y
            
            u[i][r,:]=y
        
        m=m2/float(Ntest)
        #plt.plot(np.sqrt(K/float(L))*tf.signal.idct(m,type=2,norm='ortho'))
        #plt.show()
    return u


#uncontrolled solution
def solveStateUc2(u0,w):
    
    
    #initialize
    u=np.zeros([nt,K])

    u[0,:]=u0
    
    ypre=u0
    

    for r in range(1,nt):

        
        
        y=spsolve(A,(1/float(dt))*ypre+(1/float(dt))*w[r,:])
        
        
        u[r,:]=y
        ypre=y
        
    return u


