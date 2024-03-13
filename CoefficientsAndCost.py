'''
Created on 03.11.2022

@author: alexa
'''
import numpy as np


##########################################################################################################################
############################################General Settings SPDE#########################################################
##########################################################################################################################

#boundary condition, 1 = Neumann, 0 = Dirichlet
bd=1

#domain right interval bound [0,L]
L=20

#terminal time [0,T]
T=20

##########################################################################################################################
############################################Discretization Parameters#####################################################
##########################################################################################################################

#number of Galerkin finite elements
K=400

#number of neurons uses in NN
KNN=400

#number of particles in particle system and number of particles for test simulations
N=1
Ntest=1

#mean-field interactions on=1/off=0
mf=0

#time discretization
dt=1/float(20)

#time steps
nt=int(T/float(dt))

#space grid for fourier approximation
a=np.zeros([K])

for i in range(K):
    a[i]=(2*i+1)*L/float(2*K)

nx=a.size

#gradient norm threshold
gamma=0.00001

#normalizing time 
scale=1/float(20)

#normalizing space
scale2=1

#time interval
ti=np.arange(0,T,dt)


##########################################################################################################################
############################################Coefficients General Control Problem##########################################
##########################################################################################################################

#equation: dx=[Ax+f(x)+B2*E[x]+D*g]dt+eps*BdW - stochastic control g
#cost: J(g)=E[int_0^T {l(r,x_r,L(x_r))+(1/2)||g||_H^2}dr+h(x_T,L(x_T))]

#initial condition
def uinit(y):
    
    nx=y.shape[0]
    
    u=np.zeros([nx])
    
    for i in range(nx):
        if 5<=y[i]<=15:
            u[i]=1
        
    return u

#drift coefficient for mean-field interaction
#B2=0.04*np.identity(K)
B2=0*np.identity(K)

#control coefficient '(<De_i,e_j>)_{i,j=1,...,N_h})'
D=1*np.identity(K)

#diffusion parameter
lp=0.75001
eps=0.01

#diffusion coefficient B
B=np.identity(K)
    
#Dirichlet boundary conditions
if bd==0:
    e=np.arange(1,K+1,1)
    B=np.pi**(-2*lp)*np.multiply(e**(-2*lp),B/float(L)**(-2*lp))
            
#Neumann boundary conditions  
if bd==1:
    e=np.arange(0,K,1)
    e[0]=1
    B=np.pi**(-2*lp)*np.multiply(e**(-2*lp),B/float(L)**(-2*lp))
    B[0,0]=16

#coefficient for non-linearity
q=0.5

#Nemytskii for non-linearity
def f(y):
    out=np.zeros([K])
    return out

#derivative of Nemytskii non-linearity
def df(y):
    out=np.zeros([K])
    return out

#cost coefficients
Q1=np.identity(K)
#Q2=np.identity(K)
Q2=0*np.identity(K)
R=np.identity(K)
Rinv=R
#M1=np.identity(K)
M1=np.identity(K)
#M2=np.identity(K)
M2=0*np.identity(K)

#terminal cost
def h(y,yT,Ey,EyT):
    return (1/float(2))*(M1.dot(y-yT)).dot(y-yT)+(1/float(2))*(M2.dot(Ey-EyT)).dot(Ey-EyT)

#derivative terminal cost
def dh(y,yT,Ey,EyT):
    
    return M1.dot(y-yT)+M2.dot(Ey-EyT)

#running cost
def l(r,y,yref,Ey,Eyref):
    return (1/float(2))*(Q1.dot(y-yref)).dot(y-yref)+(1/float(2))*(Q2.dot(Ey-Eyref)).dot(Ey-Eyref)

#derivative running cost
def dl(r,y,yref,Ey,Eyref):

    return Q1.dot(y-yref)+Q2.dot(Ey-Eyref)

##########################################################################################################################
#################################Coefficients LQ Control Problem##########################################################
##########################################################################################################################

#equation: dx=[Ax+B2*E[x]+Dg]dt+eps*BdW - stochastic control g
#cost: J(g)=1/2E[int_0^T (<Q1x,x>+<Q2E[x],E[x]>+<Rg,g>)dr+(<M1x_T,x_T>+<M2E[x_t],E[x_t]>)]


Rinv=R

M=np.multiply(D,np.multiply(Rinv,D))
