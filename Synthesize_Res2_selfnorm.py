#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Dec 13, 2024

Description: Synthesize the outputs from Simulation 2 with non-separable error 
in the outcome model.
"""

import numpy as np
import pandas as pd
import pickle

#=======================================================================================#

reg_mod_lst = ['LR', 'RF', 'NN']
fac_lst = [0.75, 1, 1.25, 1.5, 2, 4]
cond_type = ['true', 'kde', 'reg']
n_lst = [1000, 2000]
B = 1000

for reg in reg_mod_lst:
    for fac in fac_lst:
        for cond in cond_type:
            for n in n_lst:
                # Results for dose-response m
                m_ra5_lst = [] 
                m_ra1_lst = []
                
                m_ipw5_lst = [] 
                m_ipw1_lst = [] 
                
                m_dr5_lst = [] 
                m_sd_dr5_lst = [] 
                m_dr1_lst = [] 
                m_sd_dr1_lst = []
                for b in range(1, B+1):
                    with open('./Results/Simulation2_m_est'+str(b)+'_'+str(reg)+'_h'+str(fac)+'_condmod_'+str(cond)+'_n_'+str(n)+'_selfnorm.dat', "rb") as file:
                        m_est_ra5, m_est_ra1, m_est_ipw5, m_est_ipw1, \
                        m_est_dr5, sd_est_dr5, m_est_dr1, sd_est_dr1 = pickle.load(file)
                        
                    m_ra5_lst.append(m_est_ra5)
                    m_ra1_lst.append(m_est_ra1)
                    
                    m_ipw5_lst.append(m_est_ipw5)
                    m_ipw1_lst.append(m_est_ipw1)
                    
                    m_dr5_lst.append(m_est_dr5)
                    m_sd_dr5_lst.append(sd_est_dr5)
                    m_dr1_lst.append(m_est_dr1)
                    m_sd_dr1_lst.append(sd_est_dr1)
                    
                with open('./Syn_Results/Simulation2_m_est_'+str(reg)+'_h'+str(fac)+'_condmod_'+str(cond)+'_n_'+str(n)+'_selfnorm.dat', "wb") as file:
                    pickle.dump([m_ra5_lst, m_ra1_lst, m_ipw5_lst, m_ipw1_lst,
                                 m_dr5_lst, m_sd_dr5_lst, m_dr1_lst, m_sd_dr1_lst], file)
                        
                # Results for the derivative of dose-response curve \theta
                theta_ra5_lst = [] 
                theta_ra1_lst = []

                theta_ipw5_lst = [] 
                theta_ipw1_lst = []

                theta_dr5_lst = [] 
                theta_sd_dr5_lst = [] 
                theta_dr1_lst = [] 
                theta_sd_dr1_lst = []
                for b in range(1, B+1):
                    with open('./Results/Simulation2_theta_est'+str(b)+'_'+str(reg)+'_h'+str(fac)+'_condmod_'+str(cond)+'_n_'+str(n)+'_selfnorm.dat', "rb") as file:
                        theta_ra5, theta_ra1, theta_ipw5, theta_ipw1, \
                        theta_dr5, theta_sd5, theta_dr1, theta_sd1 = pickle.load(file)
                        
                    theta_ra5_lst.append(theta_ra5)
                    theta_ra1_lst.append(theta_ra1)
                    
                    theta_ipw5_lst.append(theta_ipw5)
                    theta_ipw1_lst.append(theta_ipw1)
                    
                    theta_dr5_lst.append(theta_dr5)
                    theta_sd_dr5_lst.append(theta_sd5)
                    theta_dr1_lst.append(theta_dr1)
                    theta_sd_dr1_lst.append(theta_sd1)
                    
                with open('./Syn_Results/Simulation2_theta_est_'+str(reg)+'_h'+str(fac)+'_condmod_'+str(cond)+'_n_'+str(n)+'_selfnorm.dat', "wb") as file:
                    pickle.dump([theta_ra5_lst, theta_ra1_lst, theta_ipw5_lst, theta_ipw1_lst,
                                 theta_dr5_lst, theta_sd_dr5_lst, theta_dr1_lst, theta_sd_dr1_lst], file)