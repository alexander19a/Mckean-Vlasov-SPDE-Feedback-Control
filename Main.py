'''
Created on 08.11.2022

@author: alexa
'''


import tensorflow as tf
import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse import spdiags
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from CoefficientsAndCost import *
from generateRef import ref
from SolveAdjoint import adjoint
from SolveSpde import solveState
from SolveAdjoint import grad
from SolveSpde import solveStateUc
import pickle
from solveRicc import Ricc, Ricc2, solveStateRiccRef
from optimization import opt

#!!First run RicSave.py before this

model = tf.keras.models.Sequential() # initialize neural network
model.add(tf.keras.layers.Input(shape=(K+1,)))
model.add(tf.keras.layers.Dense(KNN, activation='relu'))
#output layer
model.add(tf.keras.layers.Dense(K, activation='linear'))

#load old model
#model=tf.keras.models.load_model('C:/Users/alexa/eclipse-workspace/McKean-VlasovControl2/Models/modelLQBSPDE400.keras')

model.summary()



#load old gradient
#with open('grad.pickle', 'rb') as g:
#    gplot=pickle.load(g)
    
#load old gradient
#with open('L2300v8.pickle', 'rb') as g:
#    L2diff=pickle.load(g)

#load old cost
#with open('cost.pickle', 'rb') as c:
#    cost=pickle.load(c)

#with open('cost300v8.pickle', 'rb') as c2:
#    cost2=pickle.load(c2)

#with open('normalization.pickle', 'rb') as g:
#    normalize=pickle.load(g)

#solMin=normalize[0]
#solMax=normalize[1]
#normal=normalize[2]


gplot=[]
cost=[]
cost2=[]
L2diff=[]

#initial condition
u0=np.sqrt(L/float(K))*scipy.fftpack.dct(uinit(a),type=2, norm='ortho')

#reference profiles
yref=np.zeros([nt,K])
yT=yref[nt-1,:]
Eyref=yref
EyT=yT



#samples for scaling of data
#for i in range(20):
    
#    solMax=np.zeros([K])
#    solMin=20*np.ones([K])
    
    #sample signal data
#    w=(tf.random.normal([nt,K],0,dt*eps,dtype=tf.float32))
        
#    sol=solveStateUc(u0,w)
    
#    for k in range(K):
        
#        if solMin[k] > np.amin(sol[:,k]):
#            solMin[k]=np.amin(sol[:,k])
            
#        if solMax[k]< np.amax(sol[:,k]):
#            solMax[k]=np.amax(sol[:,k])


#normal=1./(solMax-solMin)

#normalization=[]
#normalization.append(solMin)
#normalization.append(solMax)
#normalization.append(normal)

#save data scaling constants
#with open('normalization.pickle', 'wb') as g:
#    pickle.dump(normalization,g, protocol=pickle.HIGHEST_PROTOCOL)

#parameters for scaling of data
solMin=0
solMax=1
normal=1

#run optimization
opt(u0,yref,yT,Eyref,EyT,model,gplot,cost,cost2,L2diff,solMin,solMax,normal)

