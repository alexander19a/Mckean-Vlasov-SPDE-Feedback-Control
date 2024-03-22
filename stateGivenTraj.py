'''
Created on Mar 12, 2020

@author: vogler
'''

import pickle
import numpy as np
import matplotlib as mlp
import sys
mlp.use('TKAgg')
from numpy import linalg as LA


def FHN2(dt,T,a,b,c,ar,ad,Tmax,Vrev,la,VT,V0,y0,w0,Iprev,J,noise,sigex):
    
    #Tid TaskId to change step size
    #g gradient
    # s initial stepsize
    #I prev contrtol (no graident applied zet)
    #T=final time
    #dt = discretization parameter for time
    #N=number of particles
    #V0,w0,y0=mean vector for initial condititon for each particle (membrane potential, recovery, conductance). initial cond will be chosen randomly normal dist 
    #a,b,c,I,sigex = parameter of non coupled FitzHugh-Nagumo model
    #J,sigJ=synaptic weights
    #Vrev,ar,ad,Tmax,la,VT=parameter for synapse
    
    M=int(T/dt) #number of time steps
    
    #covariance parameter for normal dist to chose initial conditions
    sigv=0.4
    sigw=0.4
    sigy=0.05
    
    #mean vector
    m=np.zeros([3])
    m[0]=V0
    m[1]=w0
    m[2]=y0

    #cov matrix (will be diag so components of multidim norm dist will be independent)
    cov=np.zeros([3,3])
    
    cov[0,0]=np.power(sigv,2)
    cov[1,1]=np.power(sigw,2)
    cov[2,2]=np.power(sigy,2)
    
    I=np.zeros([1,M])
    I[0,:]=Iprev[0,:]

    x=np.zeros([3,M])
    
    

    x[:,0]=m

        
    o=np.zeros([3,2])
    o[0,0]=sigex
    #calculate for each time r the state of the ith particle
    
    for r in range(1,M):        
            
            w=np.zeros([2,1])
            w[0]=noise[0,r-1]
            w[1]=noise[1,r-1]
            u=noise[2,r-1]
            
            #drift term with no interaction
            f=np.zeros([3,1])      
            f[0,0]=x[0,r-1]-(np.power(x[0,r-1],3)/3)-x[1,r-1]+I[0,r-1]
            f[1,0]=c*(x[0,r-1]+a-b*x[1,r-1])
            f[2,0]=ar*(Tmax/float(1+np.exp(-la*(x[0,r-1]-VT))))*(1-x[2,r-1])-ad*x[2,r-1]
          

            #calculation the state at time r
            for l in range(3):
                temp=x[:,r-1].reshape([3,1])+dt*f+np.sqrt(dt)*o.dot(w)
                x[l,r]=temp[l,0]  

    
    return x    