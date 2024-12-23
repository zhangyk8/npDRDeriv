#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Dec 13, 2024

Description: Synthesize the outputs from replicate experiments under the 
same setup as Simulation 2 with non-separable error in the outcome model.
"""

import numpy as np
import pandas as pd
import pickle

#=======================================================================================#

model_lst = ['knn', 'nn']
fac_lst = [0.75, 1, 1.25, 1.5, 2, 4]
n_lst = [1000, 2000, 10000]
B = 1000

for ml in model_lst:
    for fac in fac_lst:
        for n in n_lst:
            # Results for dose-response m
            m_est5_lst = [] 
            m_est1_lst = []
            
            m_std5_lst = [] 
            m_std1_lst = [] 
            
            theta_est5_lst = [] 
            theta_std5_lst = [] 
            theta_est1_lst = [] 
            theta_std1_lst = []
            for b in range(1, B+1):
                with open('./Results/Sim2_Replicate_est'+str(b)+'_'+str(ml)+'_h'+str(fac)+'_n_'+str(n)+'.dat', "rb") as file:
                    m_est1, m_std1, theta_est1, theta_std1, m_est5, m_std5, theta_est5, theta_std5 = pickle.load(file)
                    
                m_est5_lst.append(m_est5)
                m_est1_lst.append(m_est1)
                
                m_std5_lst.append(m_std5)
                m_std1_lst.append(m_std1)
                
                theta_est5_lst.append(theta_est5)
                theta_std5_lst.append(theta_std5)
                theta_est1_lst.append(theta_est1)
                theta_std1_lst.append(theta_std1)
                
            with open('./Syn_Results/Sim2_Replicate_est_'+str(ml)+'_h'+str(fac)+'_n_'+str(n)+'.dat', "wb") as file:
                pickle.dump([m_est1_lst, m_std1_lst, theta_est1_lst, theta_std1_lst, 
                             m_est5_lst, m_std5_lst, theta_est5_lst, theta_std5_lst], file)
                    