import pickle
import numpy as np
import matplotlib as mlp
import sys
mlp.use('TKAgg')
from numpy import linalg as LA


#TaskId, determines stepsize
arg = int(sys.argv[1])

#calculate solution to state eq. with control with stepsize np.power((1/2),(TaskId+1))*s

def stateStep(Tid,s,N,dt,T,a,b,c,ar,ad,Tmax,Vrev,la,VT,V0,y0,w0,Iprev,g,J,sigex,sigJ,det,det2):
    
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
    
    #update control with given step size and gradient (i.e. infinitesimal change of control at each time according to adjoint equation)
    I=np.zeros([1,M])
    I[0,:]=Iprev[0,:]-np.power((1/float(2)),(Tid))*s*g

    z=[] #list of solutions for each particle
    
    
    
    #initial conditions for each particle
    for i in range(N):     
        x=np.zeros([3,M])
        wini=np.random.multivariate_normal(m,cov)
        x[:,0]=wini
        z.append(x)
        
    
    #calculate for each time r the state of each particle
    for r in range(1,M):        
        
        m=0
        
        #calculating y-mean of prev. time
        for k in range(N):
            m=m+z[k][2,r-1]
        m=m/float(N)
        
        #state of each particle i at time r
        for i in range(N):
            w=np.random.normal(0,1,[2,1])       #FitzHugh-Nagumo noise term without coupling
            u=np.random.normal(0,1)             #synaptic noise
            
            #drift term with no interaction
            f=np.zeros([3,1])       
            f[0,0]=z[i][0,r-1]-(np.power(z[i][0,r-1],3)/3)-z[i][1,r-1]+I[0,r-1]
            f[1,0]=c*(z[i][0,r-1]+a-b*z[i][1,r-1])
            f[2,0]=ar*(Tmax/float(1+np.exp(-la*(20*z[i][0,r-1]-VT))))*(1-z[i][2,r-1])-ad*z[i][2,r-1]
            
            #mean-field drift term
            q=np.zeros([3,1])       
            
            q[0,0]=q[0,0]-J*(z[i][0,r-1]-Vrev)*m
            
            #diffusion term with no interaction
            o=np.zeros([3,2])       
            
            o[0,0]=sigex
            
            #det parameter for turning coupling noise on or off
            if z[i][2,r-1] < 1 and z[i][2,r-1] > 0 and det==0:
                #o[2,1]=sig2
                o[2,1]=(0.1*np.exp(-0.5/float(1-np.power(2*z[i][2,r-1]-1,2))))*np.sqrt(ar*(Tmax/float(1+np.exp(-la*(20*z[i][0,r-1]-VT))))*(1-z[i][2,r-1])+ad*z[i][2,r-1])
            
            #mean-field diffusion term
            h=np.zeros([3,1])       
            
            if det2==0:
                h[0,0]=h[0,0]-sigJ*(z[i][0,r-1]-Vrev)*m   

            #calculation the state at time r
            for l in range(3):
                temp=z[i][:,r-1].reshape([3,1])+dt*f+dt*q+np.sqrt(dt)*o.dot(w)+np.sqrt(dt)*u*h
                z[i][l,r]=temp[l,0]  
    return z       


#given parameter and TaskId
with open("par.pkl","rb") as f:
        z = pickle.load(f)

sig2=0

#initialize given parameter
det2=z[0]
det=z[1]
dt=z[2]
T=z[3]
N=z[4]
V0=z[5]
w0=z[6]
y0=z[7]
a=z[8]
b=z[9]
c=z[10]
Iprev=z[11]
sigex=z[12]
J=z[13]
sigJ=z[14]
Vrev=z[15]
ar=z[16]
ad=z[17]
Tmax=z[18]
la=z[19]
VT=z[20]
yT=z[21]
yR=z[22]
s=z[23]
g=z[24]
Jpre=z[25]

M=int(T/dt) 

sol=[]

#calculate solution for corresponding step size mod control
y=stateStep(arg,s,N,dt,T,a,b,c,ar,ad,Tmax,Vrev,la,VT,V0,y0,w0,Iprev,g,J,sigex,sigJ,det,det2)

#calculate new cost
Jnew=0

Inew=np.zeros([1,M])
st=np.power((1/float(2)),(arg))*s
Inew[0,:]=Iprev[0,:]-st*g

for k in range(N):
    Jnew=Jnew +dt*np.power(LA.norm(y[k][0,:]-yR[0,:]),2)+np.power(LA.norm(y[k][0,M-1]-yT[0]),2)+dt*np.power(LA.norm(Inew[0,:]),2)
Jnew = Jnew/float(N)

#if new cost smaller, return corresponding solutions and cost, else set cost to -1
#new step size


if Jpre+(1/float(100000))*st*g.dot(-g)>=Jnew:
    sol.append(y)
    sol.append(Jnew)
    sol.append(Inew)
else:
    sol.append(y)
    sol.append(-1)
    sol.append(Inew)

temp="/homes/stoch/vogler/local/state" + str(arg) + "cl=" + ".pkl" #temp save of z for the current cluster simulation

with open(temp,"wb") as f:
    pickle.dump(sol, f, pickle.HIGHEST_PROTOCOL)

print(Jpre)
print(Jnew)
