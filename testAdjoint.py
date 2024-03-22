'''
Created on Jan 8, 2020

@author: vogler
'''

import numpy as np
import matplotlib as mlp
mlp.use('TKAgg')
import matplotlib.pyplot as plt

def adjeqTest(N,T,dt,b,c,ar,ad,Tmax,la,VT,X,yT,yR,J,Vrev,mean):

    #X solution to the high dim particle system
    #I control process
    #yT terminal ref profile (V,w,y)-|(V,w,y)-yT|
    #yR running ref profile (V,w,y)-|(V,w,y)-yR[:,t]
    
    M=int(T/dt) #number of time steps

    z=[] #list adjoint particles
    
    m=np.zeros([M]) #empirical expectations of y
    
    r=0
    
    if not mean==0:
        for t in range(M):
            r=0
            for k in range(N):
                r=r+X[k][2,t]
            m[t]=r/float(N)

    
    plt.plot(m)
    plt.savefig('m.png')
    plt.close()
    plt.plot(X[0][2,:])
    plt.savefig('ref.png')
    plt.close()
 

    
    #initialize adjoint state realizations via particles
    for i in range(N):
        
        #initialize terminal condition for the ith particle (adjoint particle i depends on X[i]!)
        dgT=np.zeros([3])
        dgT[0]=2*(X[i][0,M-1]-yT[0]) #first component difference to desired terminal state (distance of its particle to the desired state...we want to control all particles)
        p=np.zeros([3,M])
        p[:,M-1]=dgT
        z.append(p)
        
    for t in range(M-2,-1,-1):  #iteration backward in time (excludes last time-2)
        
        #mean for mean field term
        r=0    
        
        if not mean==0:
            for k in range(N):
                r=r-J*((X[k][0,t+1]-Vrev)*z[k][0,t+1])

        
        for i in range(N):  #iteration over all particles
            
            #define coefficients
            
            #running cost derivative
            df=np.zeros([3])  

            df[0]=2*(X[i][0,t+1]-yR[0,t+1])    #only first component because we only want to control membrane potential

            #matrix coefficient 
            A=np.zeros([3,3])
            A[0,0]=1-np.power(X[i][0,t+1],2)-J*m[t+1]
            A[0,1]=-1
            A[1,0]=c
            A[1,1]=-c*b
            A[2,0]=ar*((la*Tmax*np.exp(-la*(X[i][0,t+1]-VT)))/float(np.power(np.exp(-la*(X[i][0,t+1]-VT))+1,2)))*(1-X[i][2,t+1])
            A[2,2]=-ad-ar*(Tmax/float(1+np.exp(-la*(X[i][0,t+1]-VT))))


            #mean field term
            phi=np.zeros([3])
        
            phi[2]=r/float(N)
    
            z[i][:,t]=z[i][:,t+1]+dt*np.transpose(A).dot(z[i][:,t+1])+dt*df+dt*phi
            
    return z 