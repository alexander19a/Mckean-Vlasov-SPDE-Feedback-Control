'''
Created on Aug 5, 2019

@author: vogler
'''


import numpy as np
import matplotlib as mlp
mlp.use('TKAgg')
import matplotlib.pyplot as plt
from singleFHN import FHN
from numpy import linalg as LA
from testAdjoint import adjeqTest
from stateGivenTraj import FHN2
#parameters for the simulation



##initial condition for each particle (parameter for initial normal distribution)
V0=0.0
w0=0.5
y0=0.3

#time discretization
dt=0.1
T=100 #final time

M=int(T/dt) 
#parameters of FitzHugh-Nagumo equation
a=0.7
b=0.8
c=0.08

sigex=0.05

#synaptic weights 
J=0 #conductance parameter for deterministic part 
    #of chemical input J=0 means no deterministic coupling
sigJ=0 #conductance parameter of stochastic part of chemical input

#synapse i.e. parameter defining reversal potential and 
#param for open channels for synaptic input

Vrev=1
ar=0
ad=0
Tmax=1
la=0.2
VT=2

#Number of neurons in the network
N=1000

#coupling channel noise and synaptic cleft and noise i.e. conductance noise
det=1
det2=1

#discretization parameter state space
dv=0.1
dw=0.1
dy=0.05
sigy=0.2

#initializing scalar control at each time
I=0.33*np.ones([1,int(T/dt)])

x = np.arange(0, 100+(100/float(1000)),100/float(999))   #time x-axis 

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

m=np.zeros([3])
sig=np.ones([3,3])

yR=np.zeros([3,M])
yT=yR[:,M-1]

state=FHN(N, dt, T, a, b, c, ar, ad, Tmax, Vrev, la, VT, V0, y0, w0, I, J, sigex, sigJ, det, det2)

mean=np.zeros([3,M])

for k in range(N):
    mean=mean+state[0][k]

mean=mean/float(N)

plt.plot(x,mean[0,:],'r')
plt.savefig('stateMean.png')
plt.close()

p=adjeqTest(N,T,dt,b,c,ar,ad,Tmax,la,VT,state[0],yT,yR,J,Vrev,0)
g=np.zeros([M])
g2=np.zeros([M])

for k in range(N):
    g[:]=g[:]+p[k][0,:] 

g=g/float(N)

for k in range(N):
    if not LA.norm(p[k][0,:],np.inf)>800:
        g2[:]=g2[:]+p[k][0,:] 

g2=g2/float(N)

#expectation adj
plt.plot(x,-g[:],'r')
plt.savefig('ad1.png')
plt.close()

plt.plot(x,-g2[:],'r')
plt.savefig('ad2.png')
plt.close()

#state trajectories
for k in range(N):
    plt.plot(x,state[0][k][0,:],'r')
plt.savefig('states.png')
plt.close()

#adjoint trajectories
for k in range(N):
    plt.plot(x,-p[k][0,:],'r')
    
plt.ylim(-200,200)
plt.savefig('adjs.png')
plt.close()

#mod control at max adj value position and plot with given noise trajectory
for k in range(N):
    I=0.33*np.ones([1,int(T/dt)])
    l=np.where(p[k][0,:] == np.amax(p[k][0,:]))
    plt.plot(x,state[0][k][0,:],'red')
    I[0,l]=I[0,l]+np.sign(p[k][0,l])*0.1
    state2=FHN2(dt,T,a,b,c,ar,ad,Tmax,Vrev,la,VT,V0,y0,w0,I,J,state[1][k],sigex)
    plt.plot(x,state2[0,:],'b')
    print(LA.norm(state[0][k][0,:]-state2[0,:],np.inf))


plt.savefig('high.png')  
plt.close()

#critical paths
crits=[]
crita=[]

for k in range(N):
    if LA.norm(p[k][0,:],np.inf)>800:
        crits.append(state[0][k][0,:])
        crita.append(-p[k][0,:])
        
for k in range(len(crits)):
    plt.plot(x,crits[k],'r')


plt.savefig('critstate.png')
plt.close()        

for k in range(len(crits)):
    plt.plot(x,crita[k],'r')
    
plt.savefig('critadj.png')
plt.close()        

