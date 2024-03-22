import numpy as np
import matplotlib as mlp
mlp.use('TKAgg')




#calculate solution to state eq. with control with stepsize np.power((1/2),(TaskId+1))*s

def stateEq(Tid,s,N,dt,T,a,b,c,ar,ad,Tmax,Vrev,la,VT,V0,y0,w0,Iprev,g,J,sigex,sigJ,sig2,det,det2):
    
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
    sigv=0.01
    sigw=0.01
    sigy=0.005
    
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
    
    #update control with given step size and gradient
    I=np.zeros([1,M])
    I[0,:]=Iprev[0,:]-np.power((1/float(2)),(Tid+1))*s*g

    z=[] #list of solutions for each particle
    
    for i in range(N):      #initial conditions for each particle
        x=np.zeros([3,M])
        wini=np.random.multivariate_normal(m,cov)
        x[:,0]=wini
        z.append(x)
        
    
    #calculate for each time r the state of the ith particle
    
    for r in range(1,M):        
        
        m=0
        #calculating mean of prev time
        for k in range(N):
            m=m+z[k][2,r-1]
        m=m/float(N)

        
        for i in range(N):
            w=np.random.normal(0,1,[2,1])       #FitzHugh-Nagumo noise term without coupling
            u=np.random.normal(0,1)             #synaptic noise
            
            f=np.zeros([3,1])       #drift term with no interaction
            f[0,0]=z[i][0,r-1]-(np.power(z[i][0,r-1],3)/3)-z[i][1,r-1]+I[0,r-1]
            f[1,0]=c*(z[i][0,r-1]+a-b*z[i][1,r-1])
            f[2,0]=ar*(Tmax/float(1+np.exp(-la*(z[i][0,r-1]-VT))))*(1-z[i][2,r-1])-ad*z[i][2,r-1]

            #mean-field drift term
            q=np.zeros([3,1])       
            
            q[0,0]=q[0,0]-J*(z[i][0,r-1]-Vrev)*m
            
            o=np.zeros([3,2])       #diffusion term with no interaction
            
            o[0,0]=sigex
            
            #det parameter for turning coupling noise on or off
            if z[i][2,r-1] < 1 and z[i][2,r-1] > 0 and det==0:
                o[2,1]=sig2
            
            h=np.zeros([3,1])       #mean-field diffusion term
            
            if det2==0:
                h[0,0]=h[0,0]-sigJ*(z[i][0,r-1]-Vrev)*m   

            #calculation the state at time r
            for l in range(3):
                temp=z[i][:,r-1].reshape([3,1])+dt*f+dt*q+np.sqrt(dt)*o.dot(w)+np.sqrt(dt)*u*h
                z[i][l,r]=temp[l,0]  
    return z       