'''
Created on Jan 9, 2020

@author: vogler
'''
import os
import pickle
import numpy as np
import matplotlib as mlp
import time
mlp.use('TKAgg')
from numpy import linalg as LA
from adjoint import adjeq
from pathlib2 import Path
import matplotlib.pyplot as plt

#parameters for the simulation


def optCl(a,b,c,ad,ar,VT,la,det,det2,sig2,Tmax,Vrev,J,sigJ,dt,T,yT,yR,N,eps,init,V0,w0,y0,sigex):
    
    #initialize for plots
    count=0
    x = np.arange(0, 100+(100/float(1000)),100/float(999))   #time x-axis 
    
    #Failmarker if no step size was accepted
    fmark=-1

    #initial step size for gradient decent
    s=float(0.02)
    
    M=int(T/dt) #number of time steps
    
    #gradient 
    g=np.zeros([M])

    #initial control
    Iprev=init

    #initial samples of state eq for intial control and 0 gradient (more b/c adj eq)
    samples=[]

    Jpre=0
    param=[]    #list of parameter for cluster simulations

    param.append(det2)
    param.append(det)       
    param.append(dt)
    param.append(T)
    param.append(N)
    param.append(V0)
    param.append(w0)
    param.append(y0)
    param.append(a)
    param.append(b)
    param.append(c)
    param.append(Iprev)
    param.append(sigex)
    param.append(J)
    param.append(sigJ)
    param.append(Vrev)
    param.append(ar)
    param.append(ad)
    param.append(Tmax)
    param.append(la)
    param.append(VT)
    param.append(yT)
    param.append(yR)
    param.append(s)
    param.append(g)
    param.append(Jpre)
                
    #dump parameters
    with open("par.pkl","wb") as f:       
        pickle.dump(param, f, pickle.HIGHEST_PROTOCOL)
        
    #mod control for 80 different step sizes and check if cost less
    os.system("qsub -t 1:30 myjob3.job")
                         
    temppath=Path("/homes/stoch/vogler/local/" + "stateSample" + str(80) + "cl=" + ".pkl")      
        
    while not temppath.is_file():       #wait until cluster is finished with computation
        time.sleep(0.5)
        print("waiting for cluster to finish")
                
    print("temp data created successfully")
                
    time.sleep(2)
                
    print("calculating solution")
                
    for r in range(10):     #go through all cluster simulations
                    
        temppath=Path("/homes/stoch/vogler/local/" + "stateSample" + str(r+1) + "cl=" + ".pkl")
            
        while not temppath.is_file():
            time.sleep(2)
                    
        time.sleep(1.1)
                    
        with open("/homes/stoch/vogler/local/" + "stateSample" + str(r+1) + "cl=" + ".pkl" ,"rb")as f:      #open data from r-th simulation
            while os.stat("/homes/stoch/vogler/local/" + "stateSample" + str(r+1) + "cl=" + ".pkl").st_size == 0:
                time.sleep(0.5)
            z = pickle.load(f)
                    
        for i in range(N):
            samples.append(z[0][i])
                    
        #remove temp files    
        os.remove("/homes/stoch/vogler/local/" + "stateSample" + str(r+1) + "cl=" + ".pkl")
    
    
    #calculate initial cost
    Jin=0
    
    #Monte Carlo for cost over all particles(samples)
    for k in range(N):
        Jin=Jin+dt*np.power(LA.norm(samples[k][0,:]-yR[0,:]),2)+np.power(LA.norm(samples[k][0,M-1]-yT[0]),2)+dt*np.power(LA.norm(Iprev[0,:]),2)
    Jpre=Jin/float(N)
    print(Jpre)
    #initial gradient

    p=adjeq(10*N,T,dt,b,c,ar,ad,Tmax,la,VT,samples,yT,yR,J,Vrev)
    
    plt.plot(x,samples[0][0,:],'r')
    plt.savefig('inistate.png')
    plt.close()
  
    
    #initial gradient

    for k in range(10*N):
        g[:]=g[:]+p[k][0,:]+2*Iprev[0,:]
            
    g[:]=g[:]/float(10*N)
    
    
    #initial gnorm
    gnorm=dt*LA.norm(g)
    
    plt.plot(x,-g[:],'r')
    plt.savefig( str(count) + 'grad2.png')
    plt.close()
    
    #break
    gnorm=0
    #while norm is beyond treshold
    
    while gnorm>eps:
        Jdiff=-1
        while Jdiff<0:
            
            plt.plot(x,-g[:],'r')
            plt.savefig( str(count) + 'grad.png')
            plt.close()
            
            param=[]    #list of parameter for cluster simulations

            param.append(det2)
            param.append(det)       
            param.append(dt)
            param.append(T)
            param.append(N)
            param.append(V0)
            param.append(w0)
            param.append(y0)
            param.append(a)
            param.append(b)
            param.append(c)
            param.append(Iprev)
            param.append(sigex)
            param.append(J)
            param.append(sigJ)
            param.append(Vrev)
            param.append(ar)
            param.append(ad)
            param.append(Tmax)
            param.append(la)
            param.append(VT)
            param.append(yT)
            param.append(yR)
            param.append(s)
            param.append(g)
            param.append(Jpre)
            
            #dump parameters
            with open("par.pkl","wb") as f:       #temp save of parameter for cluster sim
                pickle.dump(param, f, pickle.HIGHEST_PROTOCOL)
    
            #mod control for 80 different step sizes and check if cost less
            os.system("qsub -t 1-80:1 myjob2.job")
                     
            temppath=Path("/homes/stoch/vogler/local/" + "state" + str(80) + "cl=" + ".pkl")      
    
            while not temppath.is_file():       #wait until cluster is finished with computation
                    time.sleep(0.5)
                    print("waiting for cluster to finish")
            
            print("temp data created successfully")
            
            time.sleep(2)
            
            print("calculating solution")
            
            for r in range(80):     #go through all cluster simulations
                
                temppath=Path("/homes/stoch/vogler/local/" + "state" + str(r+1) + "cl=" + ".pkl")
        
                while not temppath.is_file():
                    time.sleep(2)
                
                time.sleep(1.1)
                
                with open("/homes/stoch/vogler/local/" + "state" + str(r+1) + "cl=" + ".pkl" ,"rb")as f:      #open data from r-th simulation
                    while os.stat("/homes/stoch/vogler/local/" + "state" + str(r+1) + "cl=" + ".pkl").st_size == 0:
                        time.sleep(0.5)
                    z = pickle.load(f)
                
                print(z[1])
                #looking for the smallest cost coming from some cluster
                if not z[1]==-1 and z[1]<Jpre:
                    Jpre=z[1]
                    Iprev=z[2]
                    fmark=1
                    #sol w.r.t. new control
                    y=z[0]
                    
                #remove temp files    
                os.remove("/homes/stoch/vogler/local/" + "state" + str(r+1) + "cl=" + ".pkl")     
            #if nothing is accepted, till algo
            if fmark==-1:
                print('fail')
                gnorm=-1
                break
            
            #reset
            fmark=-1
            
            samples=[]
            
            #additional samples for adj eq. b/c of high fluctuations
            if fmark==1:
                
                
                #use next control to sample
                param[11]=Iprev
                
                #dump parameters
                with open("par.pkl","wb") as f:       
                    pickle.dump(param, f, pickle.HIGHEST_PROTOCOL)
        
                #mod control for 80 different step sizes and check if cost less
                os.system("qsub -t 1-10:1 myjob3.job")
                         
                temppath=Path("/homes/stoch/vogler/local/" + "stateSample" + str(10) + "cl=" + ".pkl")      
        
                while not temppath.is_file():       #wait until cluster is finished with computation
                        time.sleep(0.5)
                        print("waiting for cluster to finish")
                
                print("temp data created successfully")
                
                time.sleep(2)
                
                print("calculating solution")
                
                for r in range(10):     #go through all cluster simulations
                    
                    temppath=Path("/homes/stoch/vogler/local/" + "stateSample" + str(r+1) + "cl=" + ".pkl")
            
                    while not temppath.is_file():
                        time.sleep(2)
                    
                    time.sleep(1.1)
                    
                    with open("/homes/stoch/vogler/local/" + "stateSample" + str(r+1) + "cl=" + ".pkl" ,"rb")as f:      #open data from r-th simulation
                        while os.stat("/homes/stoch/vogler/local/" + "stateSample" + str(r+1) + "cl=" + ".pkl").st_size == 0:
                            time.sleep(0.5)
                        z = pickle.load(f)
                    
                    for i in range(N):
                        samples.append(z[0][i])
                    
                    #remove temp files    
                    os.remove("/homes/stoch/vogler/local/" + "stateSample" + str(r+1) + "cl=" + ".pkl")
                
                print(z[1])
                #looking for the smallest cost coming from some cluster
                if not z[1]==-1 and z[1]<Jpre:
                    Jpre=z[1]
                    Iprev=z[2]
                    fmark=1
                    #sol w.r.t. new control
                    y=z[0]
                    
                #remove temp files    
                os.remove("/homes/stoch/vogler/local/" + "state" + str(r+1) + "cl=" + ".pkl")     
            
            #calculate new gradient
            g=np.zeros([M])
            p=adjeq(N,T,dt,b,c,ar,ad,Tmax,la,VT,samples,yT,yR,J,Vrev)
            
            for k in range(10*N):
                g[:]=g[:]+p[k][0,:]+2*Iprev[0,:]
                    
            g[:]=g[:]/float(10*N)
            
            gnorm=dt*LA.norm(g)
            Jdiff=0
            
            plt.plot(x,Iprev[0,:],'r')
            plt.savefig( str(count) + 'control.png')
            plt.close()
            plt.plot(x,y[0][0,:],'r')
            plt.savefig( str(count) + 'state.png')
            plt.close()
        
            count=count+1
            
    return Iprev

