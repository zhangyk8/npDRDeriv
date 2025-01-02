#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Dec 11, 2024

Description: Replicate the proposed method by Colangelo and Lee (2023) under the 
same setup as Simulation 2 with non-separable error in the outcome model.
"""

import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression  # This line is needed even we are not using LR

import Supplement
import sys
import pickle

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#


model_lst = ['knn', 'nn']
fac_lst = [0.75, 1, 1.25, 1.5, 2, 4]
# n_lst = [1000, 2000, 10000]
n_lst = [500, 4000, 6000]

for ml in model_lst:
    for fac in fac_lst:
        for n in n_lst:
            rho = 0.5  # correlation between adjacent Xs
            if n == 10000:
                d = 100   # Dimension of the confounding variables
            else:
                d = 20   # Dimension of the confounding variables


            model_nn1_n1000 = Supplement.NeuralNet1_n1000(k=d+1, 
                                                  lr=0.01,
                                                  momentum=0.9,
                                                  epochs=100,
                                                  weight_decay=0.05) 
                
            model_nn2_n1000 = Supplement.NeuralNet2_n1000(k=d, 
                                              lr=0.01,
                                              momentum = 0.9, 
                                              epochs=100,
                                              weight_decay=0.3)

            model_knn1_n1000 = Supplement.NeuralNet1k_n1000(k=d, 
                                                  lr=0.01,
                                                  momentum=0.9,
                                                  epochs=100,
                                                  weight_decay=0.05) 
                
            model_knn2_n1000 = Supplement.NeuralNet2_n1000(k=d, 
                                              lr=0.01,
                                              momentum = 0.9, 
                                              epochs=100,
                                              weight_decay=0.3)


            model_nn1_n10000 = Supplement.NeuralNet1_n10000(k=d+1, 
                                                  lr=0.05,
                                                  momentum=0.95,
                                                  epochs=100,
                                                  weight_decay=0.05) 
            model_nn2_n10000 = Supplement.NeuralNet2_n10000(k=d, 
                                                  lr=0.4,
                                                  momentum = 0.0, 
                                                  epochs=100,
                                                  weight_decay=0.075)

            model_knn1_n10000 = Supplement.NeuralNet1k_n10000(k=d, 
                                                  lr=0.05,
                                                  momentum = 0.9,
                                                  epochs=100,
                                                  weight_decay=0.1)
                
            model_knn2_n10000 = Supplement.NeuralNet2_n10000(k=d, 
                                                  lr=0.4,
                                                  momentum = 0.0, 
                                                  epochs=100,
                                                  weight_decay=0.075)
            
            
            
            # Generating the simulated data
            Sigma = np.zeros((d,d)) + np.eye(d)
            for i in range(d):
                for j in range(i+1, d):
                    if (j < i+2) or (j > i+d-2):
                        Sigma[i,j] = rho
                        Sigma[j,i] = rho
            sig = 1

            np.random.seed(job_id)
            # Data generating process
            X_sim = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
            nu = np.random.randn(n)
            eps = np.random.randn(n)

            theta = 1/(np.linspace(1, d, d)**2)

            T_sim = scipy.stats.norm.cdf(3*np.dot(X_sim, theta)) + 3*nu/4 - 1/2
            Y_sim = 1.2*T_sim + T_sim**2 + T_sim*X_sim[:,0] + 1.2*np.dot(X_sim, theta) + eps*np.sqrt(0.5+ scipy.stats.norm.cdf(X_sim[:,0]))
            X_dat = np.column_stack([T_sim, X_sim])
            
            
            # Bandwidth choice
            h = fac*np.std(T_sim)*n**(-1/5)
            eta = h*(n**(-(1/6)))
            
            t_qry = np.linspace(-2, 2, 81)
            t_qry_full = np.zeros((t_qry.shape[0], 3))
            t_qry_full[:,0] = t_qry
            t_qry_full[:,1] = t_qry - eta/2
            t_qry_full[:,2] = t_qry + eta/2
            
            
            m_est1 = np.zeros((t_qry.shape[0],))
            m_std1 = np.zeros((t_qry.shape[0],))
            theta_est1 = np.zeros((t_qry.shape[0],))
            theta_std1 = np.zeros((t_qry.shape[0],))
            
            m_est5 = np.zeros((t_qry.shape[0],))
            m_std5 = np.zeros((t_qry.shape[0],))
            theta_est5 = np.zeros((t_qry.shape[0],))
            theta_std5 = np.zeros((t_qry.shape[0],))
            
            for i in range(t_qry.shape[0]):
                t_list = t_qry_full[i,:]
                if ml=='knn':
                    if n == 10000:
                        model1 = Supplement.NN_DDMLCT(model_knn1_n10000, model_knn2_n10000)
                    else:
                        model1 = Supplement.NN_DDMLCT(model_knn1_n1000, model_knn2_n1000)
                    model1.fit(X_sim, T_sim, Y_sim, t_list, L=1, h=h, basis=False, standardize=True)
                    beta1 = model1.beta
                    std_error1 = model1.std_errors
                    
                    partial_effect1 = (beta1[2]-beta1[1])/eta
                    partial_effect_std1 = ((np.sqrt(15/6)/h)*std_error1[0])
                    
                    # L=5 cross-fitting
                    model5 = Supplement.NN_DDMLCT(model_knn1_n1000, model_knn2_n1000)
                    model5.fit(X_sim, T_sim, Y_sim, t_list, L=5, h=h, basis=False, standardize=True)
                    beta5 = model5.beta
                    std_error5 = model5.std_errors
                    
                    partial_effect5 = (beta5[2]-beta5[1])/eta
                    partial_effect_std5 = ((np.sqrt(15/6)/h)*std_error5[0])
                else:
                    if n == 10000:
                        model1 = Supplement.DDMLCT(model_nn1_n10000, model_nn2_n10000)
                    else:
                        model1 = Supplement.DDMLCT(model_nn1_n1000, model_nn2_n1000)
                    model1.fit(X_sim, T_sim, Y_sim, t_list, L=1, h=h, basis=False, standardize=True)
                    beta1 = model1.beta
                    std_error1 = model1.std_errors
                    
                    partial_effect1 = (beta1[2]-beta1[1])/eta
                    partial_effect_std1 = ((np.sqrt(5/2)/h)*std_error1[0])
                    
                    # L=5 cross-fitting
                    model5 = Supplement.DDMLCT(model_nn1_n1000, model_nn2_n1000)
                    model5.fit(X_sim, T_sim, Y_sim, t_list, L=5, h=h, basis=False, standardize=True)
                    beta5 = model5.beta
                    std_error5 = model5.std_errors
                    
                    partial_effect5 = (beta5[2]-beta5[1])/eta
                    partial_effect_std5 = ((np.sqrt(15/6)/h)*std_error5[0])
                
                m_est1[i] = beta1[0]
                m_std1[i] = std_error1[0]
                theta_est1[i] = partial_effect1
                theta_std1[i] = partial_effect_std1
                
                m_est5[i] = beta5[0]
                m_std5[i] = std_error5[0]
                theta_est5[i] = partial_effect5
                theta_std5[i] = partial_effect_std5
                
            with open('./Results/Sim2_Replicate_est'+str(job_id)+'_'+str(ml)+'_h'+str(fac)+'_n_'+str(n)+'.dat', "wb") as file:
                pickle.dump([m_est1, m_std1, theta_est1, theta_std1,
                             m_est5, m_std5, theta_est5, theta_std5], file)
        

