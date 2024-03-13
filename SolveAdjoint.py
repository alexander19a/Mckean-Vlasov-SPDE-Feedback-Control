'''
Created on 08.11.2022

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
##########################################Derive Gradient via Adjoint Calculus############################################
##########################################################################################################################

#laplacian for given boundary conditions
A=sparse.csr_matrix(laplace()+(1/float(dt))*np.identity(K))

#reduced Hamiltonian (no diffusion part due to additive noise and uncontrolled diffusion) 
#'H(t,x,p,mu,theta)'=H(t,x,p,mu,G_{model(theta)}(t,x))=<f(x)+B2*E[mu]+D*G_{model(theta)}(t,x),p>+int_[0,L] l(t,x,mu,G_{model(theta)}(t,x))dt+(1/2)*||G_{model(theta)}(t,x)||_H^2
#calculate adjoint state. equation: dp=-{Ap+H_x(t,x_t,p_t,L(x_t),G_{model(theta)}(t,x_t))+E[H_{mu}(t,X_t,P_t,L(x_t),G_{model(theta)}(t,X_t))(x_t)]}dt+QdW
def adjoint(state,model,yT,yref,Eyref,EyT,solMin,solMax,normal):
    
    
    
    #initialize adjoint
    p=[]
    
    #solution samples
    sol=state[0]

    #mean of solution
    msol=state[1]
    
    #mean
    m=np.zeros([K])
    
    for i in range(N):
        
        pi=np.zeros([nt,K])
        
        #terminal condition
        pi[nt-1,:]=dh(sol[i][nt-1,:],yT,msol[nt-1,:],EyT)
        
        p.append(pi)

        m=m+pi[nt-1,:]
    
    m=m/float(N)    
    
    for r in range(nt-2,-1,-1):
        
        m2=np.zeros([K])
        
        for i in range(N):
            
            ppre=p[i][r+1,:]
            
            #normalize input for neural network
            r1=tf.cast(ti[r]+1, tf.double) 
            y1=tf.cast(tf.reshape(tf.Variable(ppre),[K,1]),tf.double)
            x1=tf.Variable([scale2*np.multiply(sol[i][r+1,:]-solMin,normal)])
            r1=tf.Variable([[scale*r1]])
    
            #calculate H_x(t,x_t,p_t,L(x_t),G_{model(theta)}(t,x_t)) +E[H_{mu}(t,X_t,P_t,L(x_t),G_{model(theta)}(t,x_t))(x_t)],
            #where G_{model(theta)} is feedback induced by NN model and theta are NN parameters 
            with tf.GradientTape() as tape:
                
                tape.watch(x1)
                
                #evaluate control
                control=tf.cast(tf.reshape(model(tf.concat([r1,x1], axis=1)),[K,1]),tf.double)
                
                #evaluate 'control' part of Hamiltonian <D*G_{model(theta)}(t,x),p>+(1/2)*||G_{model(theta)}(t,x)||_H^2
    
                H=tf.linalg.matmul(tf.linalg.matmul(D,control),y1,transpose_a=True)+(1/float(2))*tf.norm(control)**2
                
                #differentiate
                gradients = tf.reshape(tape.gradient(H, x1),[K])
            
            #recover solution and adjoint state from prev. time step
            rsol=np.sqrt(K/float(L))*scipy.fftpack.idct(sol[i][r+1,:],type=2, norm='ortho')
            rp=np.sqrt(K/float(L))*scipy.fftpack.idct(ppre,type=2, norm='ortho')
        
            #'non-linear' part <df,p> of Hamiltonian derivative evaluated
            nonlin1=np.multiply(rp,df(rsol))
        
            #Fourier coefficients of evaluated 'non-linear' Hamiltonian derivative
            dfFourier=np.sqrt(L/float(K))*scipy.fftpack.dct(nonlin1,type=2, norm='ortho')

            #gradient of Hamiltonian
            Hx=gradients+dfFourier+dl(r+1,sol[i][r+1,:],yref[r+1,:],msol[r+1,:],Eyref[r+1,:])
            Hmu=np.transpose(B2).dot(m)

            y=spsolve(A,(1/float(dt))*ppre+Hx+Hmu)
    
            m2=m2+y
            p[i][r,:]=y
        
        
        m=m2/float(N)
        
    return p


#calculate grdient d_(theta)J(theta)=E[int_0^T H_{theta}(r,x_t,p_t,L(x_t),G_{model(theta)}(t,x_t)) dt]
def grad(p,state,model,yT,yref,Eyref,EyT,solMin,solMax,normal):
    
    #solution samples
    sol=state[0]

    #mean of solution
    msol=state[1]
    
    #cost
    c=0
    
    #output list
    ou=[]
        
    #Hamiltonian
    H=0
    
    Hmean=0
  
    
    #
    with tf.GradientTape() as tape:
        
        #calculate 'reduced' E[int_0^T H(t,x_t,p_t,L(x_t),G_{model(theta)}(t,x_t)) dt]
        for i in range(N):
            
            for r in range(nt):

                #normalize input for neural network
                y1=tf.cast(tf.reshape(tf.Variable(p[i][r,:]),[K,1]),tf.double)
                x1=tf.Variable([scale2*np.multiply(sol[i][r,:]-solMin,normal)])
                r1=tf.Variable([[tf.cast(scale*ti[r],tf.double)]])
                
                #evaluate feedback control
                control=tf.cast(tf.reshape(model(tf.concat([r1,x1], axis=1)),[K,1]),tf.double)
                    
                #calculate 'reduced' Hamiltonian
                H=H+dt*tf.linalg.matmul(tf.linalg.matmul(D,control),y1,transpose_a=True)+(1/float(2))*dt*tf.norm(control)**2
                    
                #calculate cost at time r
       
                c=c+dt*l(r,sol[i][r,:],yref[r,:],msol[r,:],Eyref[r,:])+(1/float(2))*dt*tf.norm(control)**2
            
            #cost at terminal time
            c=c+h(sol[i][nt-1,:],yT,msol[nt-1,:],EyT)
            
            Hmean=Hmean+H
        
        Hmean=Hmean/float(N)
        c=c/float(N)
        
        #derive gradient as derivative of Hamiltonian i.e. d/d(theta) E[int_0^T H(r,x_t,p_t,L(x_t),G_{model(theta)}(t,x_t)) dt]
        gr=tape.gradient(Hmean,model.trainable_variables)
        

    ou.append(gr)
    ou.append(H[0])
    ou.append(c)
    
    return ou
