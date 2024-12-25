#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Dec 22, 2024

Description: Synthesize the outputs from repeated experiments (Simulation without positivity).
"""

import numpy as np
import pickle

#=======================================================================================#


fac_lst = [0.75, 1, 1.5, 2, 3, 4, 6, 8]
n = 2000
B = 1000

# Simulation 1
for fac in fac_lst:
    # Results for the derivative of dose-response curve \theta
    theta_ra5_lst = [] 
    theta_ra1_lst = []

    theta_ipw5_lst = [] 
    theta_ipw1_lst = []

    theta_dr5_lst = [] 
    theta_sd_dr5_lst = [] 
    theta_dr1_lst = [] 
    theta_sd_dr1_lst = []
    
    theta_C_RA5_lst = [] 
    theta_C_RA1_lst = []

    theta_C_IPW5_lst = [] 
    theta_C_IPW1_lst = []

    theta_C_DR5_lst = [] 
    theta_C_sd_DR5_lst = [] 
    theta_C_DR1_lst = [] 
    theta_C_sd_DR1_lst = []
    for b in range(1, B+1):
        with open('./Results/Sim1_Nopos_theta_'+str(b)+'_h'+str(fac)+'_inner_selfnorm3.dat', "rb") as file:
            theta_ra5, theta_ra1, theta_ipw5, theta_ipw1, theta_dr5, theta_sd5, theta_dr1, theta_sd1,\
            theta_C_RA5, theta_C_RA1, theta_C_IPW5, theta_C_IPW1, theta_C_DR5, theta_C_sd5,\
            theta_C_DR1, theta_C_sd1 = pickle.load(file)
        
        # With positivity
        theta_ra5_lst.append(theta_ra5)
        theta_ra1_lst.append(theta_ra1)
        
        theta_ipw5_lst.append(theta_ipw5)
        theta_ipw1_lst.append(theta_ipw1)
        
        theta_dr5_lst.append(theta_dr5)
        theta_sd_dr5_lst.append(theta_sd5)
        theta_dr1_lst.append(theta_dr1)
        theta_sd_dr1_lst.append(theta_sd1)
        
        # Without positivity
        theta_C_RA5_lst.append(theta_C_RA5)
        theta_C_RA1_lst.append(theta_C_RA1)
        
        theta_C_IPW5_lst.append(theta_C_IPW5)
        theta_C_IPW1_lst.append(theta_C_IPW1)
        
        theta_C_DR5_lst.append(theta_C_DR5)
        theta_C_sd_DR5_lst.append(theta_C_sd5)
        theta_C_DR1_lst.append(theta_C_DR1)
        theta_C_sd_DR1_lst.append(theta_C_sd1)
        
    with open('./Syn_Results/Sim1_Nopos_theta_h'+str(fac)+'_inner_selfnorm3.dat', "wb") as file:
        pickle.dump([theta_ra5_lst, theta_ra1_lst, theta_ipw5_lst, theta_ipw1_lst,
                     theta_dr5_lst, theta_sd_dr5_lst, theta_dr1_lst, theta_sd_dr1_lst,
                     theta_C_RA5_lst, theta_C_RA1_lst, theta_C_IPW5_lst, theta_C_IPW1_lst,
                     theta_C_DR5_lst, theta_C_sd_DR5_lst, theta_C_DR1_lst, theta_C_sd_DR1_lst], file)
        


# Simulation 2
for fac in fac_lst:
    # Results for the derivative of dose-response curve \theta
    theta_ra5_lst = [] 
    theta_ra1_lst = []

    theta_ipw5_lst = [] 
    theta_ipw1_lst = []

    theta_dr5_lst = [] 
    theta_sd_dr5_lst = [] 
    theta_dr1_lst = [] 
    theta_sd_dr1_lst = []
    
    theta_C_RA5_lst = [] 
    theta_C_RA1_lst = []

    theta_C_IPW5_lst = [] 
    theta_C_IPW1_lst = []

    theta_C_DR5_lst = [] 
    theta_C_sd_DR5_lst = [] 
    theta_C_DR1_lst = [] 
    theta_C_sd_DR1_lst = []
    for b in range(1, B+1):
        with open('./Results/Sim1_Nopos_theta_'+str(b)+'_h'+str(fac)+'_noselfnorm.dat', "rb") as file:
            theta_ra5, theta_ra1, theta_ipw5, theta_ipw1, theta_dr5, theta_sd5, theta_dr1, theta_sd1,\
            theta_C_RA5, theta_C_RA1, theta_C_IPW5, theta_C_IPW1, theta_C_DR5, theta_C_sd5,\
            theta_C_DR1, theta_C_sd1 = pickle.load(file)
        
        # With positivity
        theta_ra5_lst.append(theta_ra5)
        theta_ra1_lst.append(theta_ra1)
        
        theta_ipw5_lst.append(theta_ipw5)
        theta_ipw1_lst.append(theta_ipw1)
        
        theta_dr5_lst.append(theta_dr5)
        theta_sd_dr5_lst.append(theta_sd5)
        theta_dr1_lst.append(theta_dr1)
        theta_sd_dr1_lst.append(theta_sd1)
        
        # Without positivity
        theta_C_RA5_lst.append(theta_C_RA5)
        theta_C_RA1_lst.append(theta_C_RA1)
        
        theta_C_IPW5_lst.append(theta_C_IPW5)
        theta_C_IPW1_lst.append(theta_C_IPW1)
        
        theta_C_DR5_lst.append(theta_C_DR5)
        theta_C_sd_DR5_lst.append(theta_C_sd5)
        theta_C_DR1_lst.append(theta_C_DR1)
        theta_C_sd_DR1_lst.append(theta_C_sd1)
        
    with open('./Syn_Results/Sim1_Nopos_theta_h'+str(fac)+'_noselfnorm.dat', "wb") as file:
        pickle.dump([theta_ra5_lst, theta_ra1_lst, theta_ipw5_lst, theta_ipw1_lst,
                     theta_dr5_lst, theta_sd_dr5_lst, theta_dr1_lst, theta_sd_dr1_lst,
                     theta_C_RA5_lst, theta_C_RA1_lst, theta_C_IPW5_lst, theta_C_IPW1_lst,
                     theta_C_DR5_lst, theta_C_sd_DR5_lst, theta_C_DR1_lst, theta_C_sd_DR1_lst], file)