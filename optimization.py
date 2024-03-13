'''
Created on 08.11.2022

@author: alexa
'''

import pickle 
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
from SolveSpde import *
from SolveAdjoint import grad
from solveRicc import solveStateRiccRef
import sys
from solveRicc import Ricc, Ricc2, Ricc3
np.set_printoptions(threshold=sys.maxsize)

#!!First run RicSave.py before this

##########################################################################################################################
############################################Gradient Descent Algorithm####################################################
##########################################################################################################################

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99)

#initialize Ricc solution for L^2-difference
with open('Ric.pickle', 'rb') as g:
    V=pickle.load(g)

V2=np.zeros([nt,K,K])
phi=np.zeros([nt,K])


def opt(u0,yref,yT,Eyref,EyT,model,gplot,cost,cost2,L2diff,solMin,solMax,normal):
    
    #counter for plots
    count=0
    
    #counter for plot times
    pltcount=0
    
    #default gradient norm
    gnorm=1
    
    #while norm of the gradient is larger than threshold
    while gnorm>gamma:
        
        #collect old cost and gradient plots from prev. simulations
        gradPlot=gplot
        costPlot=cost
        costPlot2=cost2
        
        print("training process:"+str(pltcount/0.2)+"%")
        
        pltcount=pltcount+1

        #sample signal data
        w=[]

        for i in range(N):
            #white noise            
            wpath=np.sqrt(dt)*eps*np.random.normal(0,1,[nt,K])
            for r in range(nt):
                wpath[r,:]=B.dot(wpath[r,:])
                
        w.append(wpath)
            
        #solve state equation for signal data
        sol=solveState(u0,model,w,solMin,solMax,normal)
            
        #solve adjoint equation for signal data
        p=adjoint(sol,model,yT,yref,Eyref,EyT,solMin,solMax,normal)
            
        #get gradient for signal data
        g=grad(p,sol,model,yT,yref,Eyref,EyT,solMin,solMax,normal)

        gr=g[0]
        gradPlot.append(LA.norm(g[1][0]))
        costPlot.append(g[2])
            
        #optimizer applies update
        optimizer.apply_gradients(zip(gr,model.trainable_variables))
        
        count=count+1
        
        
        #plot and save after 200 iterations
        if pltcount>20:
            
            print("sample test data")
            
            #sample test data
            w=[]
                
            for i in range(Ntest):            
                wpath=np.sqrt(dt)*eps*np.random.normal(0,1,[nt,K])
                for r in range(nt):
                    wpath[r,:]=B.dot(wpath[r,:])
                    
                w.append(wpath)
                
            #controlled solution
            state=solveStateTest(u0,model,w,solMin,solMax,normal)
            
            #paths of controlled solution
            sol=state[0]
            
            #mean of controlled solution
            msol=state[1]
                
            #uncontrolled solution
            soluc=solveStateUcTest(u0,w)
                
            #Ricc soluion
            solRicc=solveStateRiccRef(u0,w,V,V2,phi,yref,yT)
              
            #inverse Fourier transform for plots
            print("calculate inverse Fourier transform")
            
            #evaluated feedback control
            u=[]
                
            #evaluated adjoint
            adj1=np.zeros([nt,nx])
            
            #evaluated solution
            sol1=[]
                
            #evaluated uncontrolled solution
            sol2=[]
                
            #Ricc optimal control
            uRicc=[]
                
            #approximated control
            control=[]
                
            for i in range(Ntest):
                    
                ui=np.zeros([nt,nx])
                sol1i=np.zeros([nt,nx])
                sol2i=np.zeros([nt,nx])
                uRicci=np.zeros([nt,nx])
                controli=np.zeros([nt,nx])
                    
                u.append(ui)
                sol1.append(sol1i)
                sol2.append(sol2i)
                uRicc.append(uRicci)
                control.append(controli)

                
                
            for r in range(nt):
                adj1[r,:]=np.sqrt(K/float(L))*tf.signal.idct(p[0][r,:],type=2,norm='ortho')
                for i in range(Ntest):
                    r1=tf.cast(ti[r], tf.double) 
                    x1=tf.Variable([scale2*np.multiply(sol[i][r,:]-solMin,normal)])
                    r1=tf.Variable([[scale*r1]])
                     
                    control[i][r,:]=model(tf.concat([r1,x1], axis=1))
                        
                    u[i][r,:]=np.sqrt(K/float(L))*tf.signal.idct(control[i][r,:],type=2,norm='ortho')
                    sol1[i][r,:]=np.sqrt(K/float(L))*tf.signal.idct(sol[i][r,:],type=2,norm='ortho')
                    sol2[i][r,:]=np.sqrt(K/float(L))*tf.signal.idct(soluc[i][r,:],type=2,norm='ortho')
            
            
            print("calculate cost and L2-difference")
            
            #calculate cost and L^2-difference
            c=0
            c2=0
            ldiff=0
                
            for i in range(Ntest):
                for r in range(nt):
                    ldiff=ldiff+dt*(tf.norm(solRicc[1][i][r,:]-control[i][r,:]))**2
                    c=c+dt*l(r,sol[i][r,:],yref[r,:],msol[r,:],Eyref[r,:])+(1/float(2))*dt*tf.norm(control[i][r,:])**2
                    c2=c2+dt*l(r,solRicc[0][i][r,:],yref[r,:],msol[r,:],Eyref[r,:])+(1/float(2))*dt*tf.norm(solRicc[1][i][r,:])**2
                #cost at terminal time
                c=c+h(sol[i][nt-1,:],yT,msol[nt-1,:],EyT)
                c2=c2+h(solRicc[0][i][nt-1,:],yT,msol[nt-1,:],EyT)
            c=c/float(Ntest)
            c2=c2/float(Ntest)
            ldiff=np.sqrt(ldiff/float(Ntest))
            costPlot2.append(np.abs(c-c2))
            L2diff.append(ldiff)
            
            print("current L2diff")
            print(ldiff)
            
            print("current cost")
            print(c)
            
            #plot
            
            pltcount=0
                
            plt.plot(gradPlot)
            plt.title('current grad='+ str(g[1][0]))
            plt.xlabel('SGD iterations')
            plt.ylabel('gradient norm')
            plt.savefig('gradient')
            plt.close()
            
            plt.plot(costPlot)
            plt.title('current cost=' + str(g[2]))
            plt.xlabel('SDG iterations')
            plt.ylabel('Cost J')
            plt.savefig('cost')
            plt.close()
                
            plt.plot(costPlot2)
            plt.title('Cost J')
            plt.xlabel('SDG iterations')
            plt.ylabel('J')
            plt.savefig('cost2')
            plt.close()
                
            plt.plot(L2diff)
            plt.title('L2diff')
            plt.xlabel('SGD iterations')
            plt.ylabel('Error')
            plt.savefig('l2diff')
            plt.close()

            fig = plt.figure(figsize=plt.figaspect(0.5))
                        
            ax = fig.add_subplot(1, 1, 1, projection='3d')
                
                            
            X = np.arange(0, T,dt)
            Y = a
            Y, X = np.meshgrid(Y, X)
            Z = u[0]
            
                
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
            plt.ylabel('x')
            plt.xlabel('t')
            fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)    
                
            plt.savefig('control3D' + str(count)+ '.png')
            plt.close()
                
            fig = plt.figure(figsize=plt.figaspect(0.5))
                        
            ax = fig.add_subplot(1, 1, 1)
                
                            
            X = np.arange(0, T,dt)
            Y = a
            Y, X = np.meshgrid(Y, X)
            Z = u[0]

                
            cp = ax.contourf(X, Y, Z)
            fig.colorbar(cp) 
            plt.ylabel('x')
            plt.xlabel('t')
                
            plt.savefig('control' + str(count)+ '.png')
            plt.close()
                
            fig = plt.figure(figsize=plt.figaspect(0.5))
                
            ax = fig.add_subplot(1, 1, 1, projection='3d')
                            
            X = np.arange(0, T,dt)
            Y = a
            Y, X = np.meshgrid(Y, X)
            Z = sol2[0]
                        
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
            plt.ylabel('x')
            plt.xlabel('t')
            fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)    
                
            plt.savefig('uncontrolledSol' + str(count)+ '.png')
            plt.close()
                
            fig = plt.figure(figsize=plt.figaspect(0.5))
                
            ax = fig.add_subplot(1, 1, 1, projection='3d')
                            
            X = np.arange(0, T,dt)
            Y = a
            Y, X = np.meshgrid(Y, X)
            Z = sol1[0]
                        
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
            plt.ylabel('x')
            plt.xlabel('t')
            fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)    
                
            plt.savefig('controlledSol' + str(count)+ '.png')
            plt.close()
                
            fig = plt.figure(figsize=plt.figaspect(0.5))
                
            ax = fig.add_subplot(1, 1, 1, projection='3d')
                            
            X = np.arange(0, T,dt)
            Y = a
            Y, X = np.meshgrid(Y, X)
            Z = -adj1
                        
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
            plt.ylabel('x')
            plt.xlabel('t')
            fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)    
                
            plt.savefig('adjoint' + str(count)+ '.png')
            plt.close()
                
            #save model
                
            with open('grad.pickle', 'wb') as gc:
                pickle.dump(gradPlot,gc, protocol=pickle.HIGHEST_PROTOCOL)
                
            with open('cost.pickle', 'wb') as gc:
                pickle.dump(costPlot,gc, protocol=pickle.HIGHEST_PROTOCOL)  
                    
            with open('cost300v9.pickle', 'wb') as gc:
                pickle.dump(costPlot2,gc, protocol=pickle.HIGHEST_PROTOCOL)  
                
            with open('L2300v9.pickle', 'wb') as gc:
                pickle.dump(L2diff,gc, protocol=pickle.HIGHEST_PROTOCOL)  

                
            model.save('C:/Users/alexa/eclipse-workspace/McKean-VlasovControl2/Models/modelLQBSPDE400.keras')
