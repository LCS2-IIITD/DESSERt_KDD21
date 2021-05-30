

import pandas as pd
import numpy as np
import itertools
from numpy import isnan
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

from numpy import linalg as LA
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

import joblib
import matplotlib.backends.backend_pdf
from math import sqrt

#The Functions

#activation function

def Activation_function(Din):
    m,n =Din.shape
    eta=0.99
    u,s,v=np.linalg.svd(Din)
    B = np.zeros(Din.shape)
    for i in range(m) :
        for j in range(n) :
            if i == j : B[i,j] = s[j]
    for i in range(m):
        for j in range(n):
            if i==j:B[i,j]=min(B[i,j],eta)
    Dout=np.dot(u,np.dot(B,v))
    return Dout

#Metric functions

def getRMSEForThreads(PredictedMatrix,ActualMatrix,TC):
  List=[]
  for i in range(0,TC):
    List.append(sqrt(mean_squared_error(list(ActualMatrix[i]),list(PredictedMatrix[i]))))
  return List

def getMaeForThreads(PredictedMatrix,ActualMatrix,TC):
  List=[]
  for i in range(0,TC):
    List.append(mean_absolute_error(list(ActualMatrix[i]),list(PredictedMatrix[i])))
  return List

#Update the predict Matrix as prediction moves

def updatePredictedMatrix(predicted,pred_day_start,pred_day_end,PredictedMatrix,window_size):
  PredictedMatrix[:,(pred_day_start-window_size):(pred_day_end-window_size)]=predicted
  return PredictedMatrix

#Prepare data for input to model

def AppendFuturePrediction(Thread,Window_Size):
  WholeDataMatrix=Thread
  X_row,X_col=WholeDataMatrix.shape
  #append shifted matrix along rows 
  WholeDataMatrix_shifted=WholeDataMatrix[:,1:X_col]
  observation_data=np.concatenate((WholeDataMatrix[:,0:(X_col-1)],WholeDataMatrix_shifted[0].reshape((1,(X_col-1)))), axis=0)
  X_row,X_col=observation_data.shape

  return X_row,X_col,WholeDataMatrix,observation_data

#The model

def FindHyperparameters(sigma,sigma1,sigma2,f1,f2,f3,thread_count,seed1,seed2,Nz,Nx,intialisationFactor,FuturePredDays,no_iter,day,pred_day_start,pred_day_end,window_size,WholeDataMatrix,observation_data,X_col):
  WrongSeedFlag=1
  rows,cols=WholeDataMatrix.shape
  
  PredictedMatrix=np.zeros((1,(FuturePredDays+3)))
  ActualMatrix=WholeDataMatrix[0,window_size+1:(window_size+FuturePredDays+4)].reshape((1,(FuturePredDays+3)))
 
  TC=1
  # D1 as A0A1A2 . Intialize there dimensions
  A_00 = Nz
  A_01 = A_00
  A_10 = A_01
  A_11 = A_10
  A_20 = A_11
  A_21 = A_20

 
  H_00 = Nx+TC
  H_01 = Nz
  H_10 = H_01
  H_11 = H_10
  H_20 = H_11
  H_21 = Nz
# Random Initialisation for all the dictionaries 
  A_0= np.random.rand(A_00,A_00) 
  A_1= np.random.rand(A_00,A_00) 
  A_2= np.eye(A_20)   #we wont estimate it, and set it to identity 
  H_0= np.random.rand(H_00,H_01) 
  H_1= np.random.rand(H_10,H_11) 
  H_2= np.eye(H_20) #we wont estimate it, and set it to identity 
  
#Flags for check
  is_A_0 = 1
  is_A_1 = 1
  is_A_2 = 0
  is_H_0 = 1
  is_H_1 = 1
  is_H_2 = 0

  #initialising 
  if (is_A_0==1):
    A_0_est = intialisationFactor*np.eye(A_00)
  else:
    A_0_est = A_0

  if (is_A_1==1):
    A_1_est = intialisationFactor*np.eye(A_10)
  else:
    A_1_est = A_1

  if (is_A_2==1):
    A_2_est = intialisationFactor*np.eye(A_20)
  else:
    A_2_est = A_2
  np.random.seed(seed1)
  if (is_H_0==1):
    H_0_est = np.random.rand(H_00,H_01)
  else:
    H_0_est = H_0    
  np.random.seed(seed2)
  if (is_H_1==1):
    H_1_est = np.random.rand(H_10,H_11)
  else:
    H_1_est = H_1
  if (is_H_2==1):
    H_2_est = np.random.rand(H_20,H_21)
  else:
    H_2_est = H_2  
        

  D_1_init = f1*np.dot(np.dot(A_0_est,A_1_est),A_2_est)
  D_2_init = f2*np.dot(np.dot(H_0_est,H_1_est),H_2_est)
        
  D_1_est_a_ = D_1_init
  D_2_est_a_ = D_2_init  

  count1=0
  num=pred_day_end+3
  
  while(pred_day_end<=num):
            window_data=observation_data[:,day:pred_day_start]
            next_seven_day_price=observation_data[:,pred_day_start:pred_day_end]
  
            #onlineRDL
            #Initialising D1, D2
        
        
            rows,cols= window_data.shape
            kalman_mean_a =np.zeros((Nz,cols))  
            kalman_covariance_a =np.zeros((cols,Nz,Nz))
            kalman_covariance_a[0] = (sigma*sigma) * np.eye(Nz)

            Q = (sigma1*sigma1) * np.eye(Nz)
            #TC row appended
            R = (sigma2*sigma2) * np.eye(Nx+TC)
            Q_inv = np.linalg.inv(Q)
            R_inv = np.linalg.inv(R)
            for i in range(0,Nz):
                kalman_mean_a[i][0]=window_data[i][0]

            smooth_mean_a =np.zeros((Nz,cols))                 
            smooth_covariance_a =np.zeros((cols,Nz,Nz))        
            G =np.zeros((cols,Nz,Nz))
            
            # worm start   
            if(count1 % 100==0):
                #reinitialising with random value after every 50 days to avoid -ve weights'
                if (is_A_0==1):
                    A_0_est = intialisationFactor*np.eye(A_00)
                else:
                    A_0_est = A_0

                if (is_A_1==1):
                    A_1_est = intialisationFactor*np.eye(A_10)
                else:
                    A_1_est = A_1

                if (is_A_2==1):
                    A_2_est = intialisationFactor*np.eye(A_20)
                else:
                    A_2_est = A_2
                np.random.seed(seed1)
                if (is_H_0==1):
                    H_0_est = np.random.rand(H_00,H_01)
                else:
                    H_0_est = H_0    
                np.random.seed(seed2)
                if (is_H_1==1):
                    H_1_est = np.random.rand(H_10,H_11)
                else:
                    H_1_est = H_1
                if (is_H_2==1):
                    H_2_est = np.random.rand(H_20,H_21)
                else:
                    H_2_est = H_2  


                D_1_init = f1*np.dot(np.dot(A_0_est,A_1_est),A_2_est) #D1
                D_2_init = f2*np.dot(np.dot(H_0_est,H_1_est),H_2_est) #D2
                D_1_est_a_ = D_1_init
                D_2_est_a_ = D_2_init
           
            for n in range(0,no_iter):
                for k in range(1,cols):
                    prior_m=np.dot(D_1_est_a_, kalman_mean_a[:,k-1])
                    prior_p=np.dot((np.dot(D_1_est_a_,kalman_covariance_a[k-1])),D_1_est_a_.T)+Q
                    y=window_data[:,k]-np.dot(D_2_est_a_,prior_m)
                    try:
                      s=np.dot(np.dot(D_2_est_a_,prior_p),D_2_est_a_.T)+R
                      K=np.dot(np.dot(prior_p,D_2_est_a_.T),np.linalg.pinv(s))
                    except:
                    	continue
                    kalman_mean_a[:,k]=  prior_m + np.dot(K,y)
                    kalman_covariance_a[k]=prior_p-np.dot(np.dot(K,s),K.T)

                smooth_mean_a[:,cols-1]= kalman_mean_a[:,cols-1]
                smooth_covariance_a[cols-1]=kalman_covariance_a[cols-1]

                for k in range(cols-2,-1,-1):
                    post_m=np.dot(D_1_est_a_, kalman_mean_a[:,k])
                    post_p=np.dot(np.dot(D_1_est_a_,kalman_covariance_a[k]),D_1_est_a_.T) +Q
                    try:
                        G[k]=np.dot(np.dot(kalman_covariance_a[k],D_1_est_a_.T),np.linalg.pinv(post_p))
                    except:
                        continue
                    smooth_mean_a[:,k]=kalman_mean_a[:,k]+np.dot(G[k],(smooth_mean_a[:,k+1]-post_m))
                    smooth_covariance_a[k]=kalman_covariance_a[k]+np.dot(np.dot(G[k],(smooth_covariance_a[k+1]-post_p)),G[k].T)

                
                Sigma = np.zeros((Nz,Nz))
                Phi   = np.zeros((Nz,Nz))
                B     = np.zeros((Nx+TC,Nz))
                C     = np.zeros((Nz,Nz))
                delta = np.zeros((Nx+TC,Nx+TC))


                for k in range(1,cols):
                    X_k = window_data[:,k]
                    X_k = X_k[:, np.newaxis] 

                    ms_k = smooth_mean_a[:,k]

                    ms_k = ms_k[:, np.newaxis]        

                    ms_k_old = smooth_mean_a[:,k-1]
                    ms_k_old = ms_k_old[:, np.newaxis] 

                    Sigma = Sigma + 1/cols * (smooth_covariance_a[k] + np.dot(ms_k,ms_k.T))
                    Phi   = Phi   + 1/cols * (smooth_covariance_a[k-1] + np.dot(ms_k_old,ms_k_old.T))
                    B     = B     + 1/cols * (np.dot(X_k,ms_k.T))
                    C     = C     + 1/cols * (np.dot(smooth_covariance_a[k],G[k-1].T) + np.dot(ms_k,ms_k_old.T))
                    delta = delta + 1/cols * (np.dot(X_k,X_k.T))
                
               
                
                if is_A_0==1 :
                    try:
                        temp = np.linalg.pinv(np.dot(np.dot(np.dot(A_1_est,A_2_est),Phi),np.dot(A_2_est.T,A_1_est.T)))           
                        A_0_est = Activation_function(np.dot(np.dot(C,np.dot(A_2_est.T,A_1_est.T)),temp))
                    except:
                        continue

                if is_A_1==1 :
                    try:
                        temp0 = np.linalg.pinv(np.dot(np.dot(A_0_est.T,Q_inv),A_0_est))
                        temp2 = np.linalg.pinv(np.dot(np.dot(A_2_est,Phi),A_2_est.T))           
                        A_1_est= Activation_function(np.dot(np.dot(np.dot(np.dot(np.dot(temp0,A_0_est.T),Q_inv),C),A_2_est.T),temp2))
                    except:
                        continue
                        
                if (is_A_2==1):
                    try:
                        temp = np.linalg.pinv(np.dot(np.dot(np.dot(np.dot(A_1_est.T,A_0_est.T),Q_inv),A_0_est),A_1_est))           
                        A_2_est= Activation_function(np.dot(np.dot(np.dot(np.dot(temp,A_1_est.T),A_0_est.T),Q_inv), np.dot(C,np.linalg.pinv(Phi))))
                    except:
                        continue
                   
                if (is_H_0==1):
                    try:
                        tempH0 = np.linalg.pinv(np.dot(np.dot(np.dot(H_1_est,H_2_est),Sigma),np.dot(H_2_est.T,H_1_est.T)))                      
                        H_0_est=np.dot(np.dot(B,np.dot(H_2_est.T,H_1_est.T)),tempH0)
                    except:
                        continue
                        
                if (is_H_1==1): 
                    try:
                        tempH1 = np.linalg.pinv(np.dot(np.dot(H_0_est.T,R_inv),H_0_est))
                        tempH2 = np.linalg.pinv(np.dot(np.dot(H_2_est,Sigma),H_2_est.T))
                        H_1_est=np.dot(np.dot(np.dot(np.dot(np.dot(tempH1,H_0_est.T),R_inv),B),H_2_est.T),tempH2)
                    except:
                        continue
                   
                if (is_H_2==1):
                    try:
                        tempH3 = np.linalg.pinv(np.dot(np.dot(np.dot(np.dot(H_1_est.T,H_0_est.T),R_inv),H_0_est),H_1_est))
                        H_2_est= np.dot(np.dot(np.dot(np.dot(tempH3,H_1_est.T),H_0_est.T),R_inv),np.dot(B,np.linalg.pinv(Sigma)))
                    except: 
                        continue
            

                
                #updated D1 and D2 from each iteration 
                D_1_est_a_new= (np.dot(np.dot(A_0_est,A_1_est),A_2_est))
                D_2_est_a_new= np.dot(np.dot(H_0_est,H_1_est),H_2_est)
                
                D_1_est_a_=D_1_est_a_new
                D_2_est_a_=D_2_est_a_new
            
            #predicting for tomorrow            
            D_1=D_1_est_a_
            D_2=D_2_est_a_[0:Nx,:]
            W_T=D_2_est_a_[Nx:,:]
            col= FuturePredDays 

            kalman_mean_test =np.zeros((Nz,col+1)) 

            kalman_mean_test[:,0:1]=kalman_mean_a[:,cols-1].reshape(Nz,1)
            kalman_covariance_test =np.zeros((col,Nz,Nz))
            kalman_covariance_test[0] = (sigma*sigma) * np.eye(Nz)

            Q = (sigma1*sigma1) * np.eye(Nz)
            R = (sigma2*sigma2) * np.eye(Nx)
        
            for k in range(1,col+1):
                kalman_mean_test[:,k]=np.dot(D_1, kalman_mean_test[:,k-1])
                

            predicted=np.dot(W_T,kalman_mean_test[:,1:])
            PredictedMatrix=updatePredictedMatrix(predicted,pred_day_start,pred_day_end,PredictedMatrix,window_size)
            day+=1
            pred_day_start=window_size+day        
            pred_day_end=pred_day_start + FuturePredDays
            count1+=1

  if not WrongSeedFlag:
    return 10,10
    
  x=list(range(window_size,num)) 
  PredictedMatrix[np.isnan(PredictedMatrix)]=0
  MAE_List=getMaeForThreads(PredictedMatrix,ActualMatrix,TC)
  RMSE_List=getRMSEForThreads(PredictedMatrix,ActualMatrix,TC)         
  return MAE_List[0],RMSE_List[0]

def ForecastForThread(sigma,sigma1,sigma2,f1,f2,f3,thread_count,seed1,seed2,Nz,Nx,intialisationFactor,FuturePredDays,no_iter,day,pred_day_start,pred_day_end,window_size,WholeDataMatrix,observation_data,X_col):

  rows,cols=WholeDataMatrix.shape
  PredictedMatrix=np.zeros((1,(cols-(window_size+1))))
  ActualMatrix=WholeDataMatrix[0,window_size+1:].reshape((1,(cols-(window_size+1))))
  TC=1
  A_00 = Nz
  A_01 = A_00
  A_10 = A_01
  A_11 = A_10
  A_20 = A_11
  A_21 = A_20

  H_00 = Nx+TC # extra one is weights for w that we wish to learn for close price.
  H_01 = Nz
  H_10 = H_01
  H_11 = H_10
  H_20 = H_11
  H_21 = Nz
# Random Initialisation for all the dictionaries 
  A_0= np.random.rand(A_00,A_00) 
  A_1= np.random.rand(A_00,A_00) 
  A_2= np.eye(A_20)   #we wont estimate it, and set it to identity 
  H_0= np.random.rand(H_00,H_01) 
  H_1= np.random.rand(H_10,H_11) 
  H_2= np.eye(H_20) #we wont estimate it, and set it to identity 

  
#Flags for check
  is_A_0 = 1
  is_A_1 = 1
  is_A_2 = 0 #we wont estimate it as 2 layer
  is_H_0 = 1
  is_H_1 = 1
  is_H_2 = 0 #we wont estimate it as 2 layer

  #initialising 
  if (is_A_0==1):
    A_0_est = intialisationFactor*np.eye(A_00)
  else:
    A_0_est = A_0

  if (is_A_1==1):
    A_1_est = intialisationFactor*np.eye(A_10)
  else:
    A_1_est = A_1

  if (is_A_2==1):
    A_2_est = intialisationFactor*np.eye(A_20)
  else:
    A_2_est = A_2
  np.random.seed(seed1)
  if (is_H_0==1):
    H_0_est = np.random.rand(H_00,H_01)
  else:
    H_0_est = H_0    
  np.random.seed(seed2)
  if (is_H_1==1):
    H_1_est = np.random.rand(H_10,H_11)
  else:
    H_1_est = H_1
  if (is_H_2==1):
    H_2_est = np.random.rand(H_20,H_21)
  else:
    H_2_est = H_2  
        

  D_1_init = f1*np.dot(np.dot(A_0_est,A_1_est),A_2_est)
  D_2_init = f2*np.dot(np.dot(H_0_est,H_1_est),H_2_est)
        
  D_1_est_a_ = D_1_init
  D_2_est_a_ = D_2_init


  count1=0
  pe=pred_day_end
  ps=pred_day_start
  while(pe<=X_col):
    day+=1
    count1+=1
    ps=window_size+day        
    pe=ps +FuturePredDays
  day=0

  Predicted_values=np.zeros((count1,FuturePredDays))
  Kalman_mean_test_save=np.zeros((count1,(FuturePredDays*5)))
  count1=0
  while(pred_day_end+1<=(X_col+1)):
            #print('Day:',day,' pred_start:',pred_day_start,' pred_end:',pred_day_end)
            window_data=observation_data[:,day:pred_day_start]
  
            #onlineRDL
            #Initialising D1, D2
        
        
            rows,cols= window_data.shape
            kalman_mean_a =np.zeros((Nz,cols))  
            kalman_covariance_a =np.zeros((cols,Nz,Nz))
            kalman_covariance_a[0] = (sigma*sigma) * np.eye(Nz)

            Q = (sigma1*sigma1) * np.eye(Nz)
            #TC row appended
            R = (sigma2*sigma2) * np.eye(Nx+TC)
            Q_inv = np.linalg.inv(Q)
            R_inv = np.linalg.inv(R)

            for i in range(0,Nz):
                kalman_mean_a[i][0]=window_data[i][0]

            smooth_mean_a =np.zeros((Nz,cols))                 
            smooth_covariance_a =np.zeros((cols,Nz,Nz))        
            G =np.zeros((cols,Nz,Nz))
            
            # worm start   
            if(count1 % 100==0):
                if (is_A_0==1):
                    A_0_est = intialisationFactor*np.eye(A_00)
                else:
                    A_0_est = A_0

                if (is_A_1==1):
                    A_1_est = intialisationFactor*np.eye(A_10)
                else:
                    A_1_est = A_1

                if (is_A_2==1):
                    A_2_est = intialisationFactor*np.eye(A_20)
                else:
                    A_2_est = A_2
                np.random.seed(seed1)
                if (is_H_0==1):
                    H_0_est = np.random.rand(H_00,H_01)
                else:
                    H_0_est = H_0    
                np.random.seed(seed2)
                if (is_H_1==1):
                    H_1_est = np.random.rand(H_10,H_11)
                else:
                    H_1_est = H_1
                
                if (is_H_2==1):
                    H_2_est = np.random.rand(H_20,H_21)
                else:
                    H_2_est = H_2  


                D_1_init = f1*np.dot(np.dot(A_0_est,A_1_est),A_2_est) #D1
                D_2_init = f2*np.dot(np.dot(H_0_est,H_1_est),H_2_est) #D2
                D_1_est_a_ = D_1_init
                D_2_est_a_ = D_2_init

            for n in range(0,no_iter):
                for k in range(1,cols):
                    prior_m=np.dot(D_1_est_a_, kalman_mean_a[:,k-1])
                    prior_p=np.dot((np.dot(D_1_est_a_,kalman_covariance_a[k-1])),D_1_est_a_.T)+Q
                    y=window_data[:,k]-np.dot(D_2_est_a_,prior_m)
                    try:
                      s=np.dot(np.dot(D_2_est_a_,prior_p),D_2_est_a_.T)+R
                      K=np.dot(np.dot(prior_p,D_2_est_a_.T),np.linalg.pinv(s))
                    except:
                    	continue
                    kalman_mean_a[:,k]=  prior_m + np.dot(K,y)
                    kalman_covariance_a[k]=prior_p-np.dot(np.dot(K,s),K.T)

                smooth_mean_a[:,cols-1]= kalman_mean_a[:,cols-1]
                smooth_covariance_a[cols-1]=kalman_covariance_a[cols-1]

                for k in range(cols-2,-1,-1):
                    post_m=np.dot(D_1_est_a_, kalman_mean_a[:,k])
                    post_p=np.dot(np.dot(D_1_est_a_,kalman_covariance_a[k]),D_1_est_a_.T) +Q
                    try:
                        G[k]=np.dot(np.dot(kalman_covariance_a[k],D_1_est_a_.T),np.linalg.pinv(post_p))
                    except:
                        continue
                    smooth_mean_a[:,k]=kalman_mean_a[:,k]+np.dot(G[k],(smooth_mean_a[:,k+1]-post_m))
                    smooth_covariance_a[k]=kalman_covariance_a[k]+np.dot(np.dot(G[k],(smooth_covariance_a[k+1]-post_p)),G[k].T)

               
                # Dictionary update : M-step
                Sigma = np.zeros((Nz,Nz))
                Phi   = np.zeros((Nz,Nz))
                B     = np.zeros((Nx+TC,Nz))
                C     = np.zeros((Nz,Nz))
                delta = np.zeros((Nx+TC,Nx+TC))


                for k in range(1,cols):
                    X_k = window_data[:,k]
                    X_k = X_k[:, np.newaxis] 

                    ms_k = smooth_mean_a[:,k]

                    ms_k = ms_k[:, np.newaxis]        

                    ms_k_old = smooth_mean_a[:,k-1]
                    ms_k_old = ms_k_old[:, np.newaxis] 

                    Sigma = Sigma + 1/cols * (smooth_covariance_a[k] + np.dot(ms_k,ms_k.T))
                    Phi   = Phi   + 1/cols * (smooth_covariance_a[k-1] + np.dot(ms_k_old,ms_k_old.T))
                    B     = B     + 1/cols * (np.dot(X_k,ms_k.T))
                    C     = C     + 1/cols * (np.dot(smooth_covariance_a[k],G[k-1].T) + np.dot(ms_k,ms_k_old.T))
                    delta = delta + 1/cols * (np.dot(X_k,X_k.T))
                
                
                if is_A_0==1 :
                    try:
                        temp = np.linalg.pinv(np.dot(np.dot(np.dot(A_1_est,A_2_est),Phi),np.dot(A_2_est.T,A_1_est.T)))           
                        A_0_est = Activation_function(np.dot(np.dot(C,np.dot(A_2_est.T,A_1_est.T)),temp))
                    except:
                        continue

                if is_A_1==1 :
                    try:
                        temp0 = np.linalg.pinv(np.dot(np.dot(A_0_est.T,Q_inv),A_0_est))
                        temp2 = np.linalg.pinv(np.dot(np.dot(A_2_est,Phi),A_2_est.T))           
                        A_1_est= Activation_function(np.dot(np.dot(np.dot(np.dot(np.dot(temp0,A_0_est.T),Q_inv),C),A_2_est.T),temp2))
                    except:
                        continue
                        
                if (is_A_2==1):
                    try:
                        temp = np.linalg.pinv(np.dot(np.dot(np.dot(np.dot(A_1_est.T,A_0_est.T),Q_inv),A_0_est),A_1_est))           
                        A_2_est= Activation_function(np.dot(np.dot(np.dot(np.dot(temp,A_1_est.T),A_0_est.T),Q_inv), np.dot(C,np.linalg.pinv(Phi))))
                    except:
                        continue
                   
                if (is_H_0==1):
                    try:
                        tempH0 = np.linalg.pinv(np.dot(np.dot(np.dot(H_1_est,H_2_est),Sigma),np.dot(H_2_est.T,H_1_est.T)))                      
                        H_0_est=np.dot(np.dot(B,np.dot(H_2_est.T,H_1_est.T)),tempH0)
                    except:
                        continue
                        
                if (is_H_1==1): 
                    try:
                        tempH1 = np.linalg.pinv(np.dot(np.dot(H_0_est.T,R_inv),H_0_est))
                        tempH2 = np.linalg.pinv(np.dot(np.dot(H_2_est,Sigma),H_2_est.T))
                        H_1_est=np.dot(np.dot(np.dot(np.dot(np.dot(tempH1,H_0_est.T),R_inv),B),H_2_est.T),tempH2)
                    except:
                        continue
                   
                if (is_H_2==1):
                    try:
                        tempH3 = np.linalg.pinv(np.dot(np.dot(np.dot(np.dot(H_1_est.T,H_0_est.T),R_inv),H_0_est),H_1_est))
                        H_2_est= np.dot(np.dot(np.dot(np.dot(tempH3,H_1_est.T),H_0_est.T),R_inv),np.dot(B,np.linalg.pinv(Sigma)))
                    except: 
                        continue
            

                
                #updated D1 and D2 from each iteration 
                D_1_est_a_new= (np.dot(np.dot(A_0_est,A_1_est),A_2_est))
                D_2_est_a_new= np.dot(np.dot(H_0_est,H_1_est),H_2_est)
               
                D_1_est_a_=D_1_est_a_new
                D_2_est_a_=D_2_est_a_new
           
            #predicting for tomorrow            
            D_1=D_1_est_a_
            D_2=D_2_est_a_[0:Nx,:]
            W_T=D_2_est_a_[Nx:,:]
            col= FuturePredDays

            kalman_mean_test =np.zeros((Nz,col+1)) 
            kalman_mean_test[:,0:1]=kalman_mean_a[:,cols-1].reshape(Nz,1)
            kalman_covariance_test =np.zeros((col,Nz,Nz))
            kalman_covariance_test[0] = (sigma*sigma) * np.eye(Nz)

            Q = (sigma1*sigma1) * np.eye(Nz)
            R = (sigma2*sigma2) * np.eye(Nx)
        
            for k in range(1,col+1):
                kalman_mean_test[:,k]=np.dot(D_1, kalman_mean_test[:,k-1])

            predicted=np.dot(W_T,kalman_mean_test[:,1:])

            #added
            Predicted_values[count1,:]=predicted.reshape(FuturePredDays,)
            Kalman_mean_test_save[count1,:] = kalman_mean_test[:,1:].reshape(col*5,)
            
            PredictedMatrix=updatePredictedMatrix(predicted,pred_day_start,pred_day_end,PredictedMatrix,window_size)
            day+=1
            pred_day_start=window_size+day        
            pred_day_end=pred_day_start + FuturePredDays
            count1+=1

  x=list(range(window_size+1,(X_col+1))) 
  for i in range(0,TC):
    fig=plt.figure()
    plt.plot(x,list(PredictedMatrix[i]), 'r') 
    plt.plot(x,ActualMatrix[i], 'b')
    plt.title('thread'+str(thread_count))
    plt.xlabel('Window Frame')
    plt.ylabel('HateScore')
    plt.legend(["predicted", "Actual"], loc ="lower right")
    plt.show()
    pdfForFeature.savefig(fig)

#load dataset

import pickle
with open('time_series.pkl', 'rb') as f:
    data = pickle.load(f)

def CreateTestSet(DataList):
  TheTestList=[]
  TestThreadName=[]
  for data in DataList:
    TotalLength=len(data[0])
    Data=np.concatenate((data[0].reshape((1,TotalLength)), data[1].reshape((1,TotalLength)),data[2].reshape((1,TotalLength)),data[3].reshape((1,TotalLength)),data[4].reshape((1,TotalLength))), axis=0)
    rows,cols=Data.shape
    trainlen=int(0.7*cols)
    trainlen=trainlen-20
    if trainlen >0 :
      TestThreadName.append(data[5][:-4])
      TheTestList.append(Data[:,trainlen:])
      rows,cols=Data[:,trainlen:].shape
  return TheTestList,TestThreadName
DataList,TestThreadName=CreateTestSet(data)

HyperParameters={
    'seed1':[6,7,8,9],
    'seed2':[6,7,8,9],
    'Nz':[5],
    'intialisationFactor':[0.5,0.7,0.9],
    'FuturePredDays':[1],
    'no_iter':[20],
    'f1':[0.5],
    'f2':[0.5],
    'f3':[0.5],
    'sigma':[0.00001],
    'sigma1':[0.1],
    'sigma2':[0.1],
    'window_size':[20]
}

pdfForFeature = matplotlib.backends.backend_pdf.PdfPages("./temp3.pdf")
counter=1
FinalMetricDict={}
for Thread in DataList:
  MetricDict={}
  HPcount=0
  HyperParameterEachThread=[]
  keys = list(HyperParameters)
  for values in itertools.product(*map(HyperParameters.get, keys)):
    HP=dict(zip(keys, values))
    HyperParameterEachThread.append(HP)
    day=0
    window_size=HP['window_size']
    pred_day_start= window_size+day
    pred_day_end=pred_day_start +HP['FuturePredDays']
    X_row,X_col,WholeDataMatrix,observation_data = AppendFuturePrediction(Thread,window_size)
    #get best fit hyperparameters
    Thread_MAE,Thread_RMSE= FindHyperparameters(HP['sigma'],HP['sigma1'],HP['sigma2'],HP['f1'],HP['f2'],HP['f3'],counter,HP['seed1'],HP['seed2'],HP['Nz'],5,HP['intialisationFactor'],HP['FuturePredDays'],HP['no_iter'],day,pred_day_start,pred_day_end,window_size,WholeDataMatrix,observation_data,X_col)
    MetricDict[HPcount]={'MAE':Thread_MAE,'RMSE':Thread_RMSE}
    HPcount=HPcount+1
  df=pd.DataFrame.from_dict(MetricDict, orient='index')
  df=df.sort_values(by='RMSE')
  ArgIndex=int(df.index[0])
  HP=HyperParameterEachThread[ArgIndex]
  day=0
  window_size=HP['window_size']
  pred_day_start= window_size+day
  pred_day_end=pred_day_start +HP['FuturePredDays']
  #get xk
  X_row,X_col,WholeDataMatrix,observation_data = AppendFuturePrediction(Thread,window_size)
  #get values after prediction
  Thread_MAE,Thread_RMSE,PredictedMatrix,ActualMatrix= ForecastForThread(HP['sigma'],HP['sigma1'],HP['sigma2'],HP['f1'],HP['f2'],HP['f3'],counter,HP['seed1'],HP['seed2'],HP['Nz'],5,HP['intialisationFactor'],HP['FuturePredDays'],HP['no_iter'],day,pred_day_start,pred_day_end,window_size,WholeDataMatrix,observation_data,X_col)
  FinalMetricDict[counter]={'MAE':Thread_MAE,'RMSE':Thread_RMSE}
  df=pd.DataFrame(ActualMatrix[0].tolist(),columns=['original'])
  df['predicted']=PredictedMatrix[0].tolist()
  df.to_csv('./'+TestThreadName[ counter-1]+"_DSS.csv")
  counter=counter+1 

pdfForFeature.close()
FinalThreadWiseScore=pd.DataFrame.from_dict(FinalMetricDict, orient='index')
print(FinalThreadWiseScore.mean(axis = 0))