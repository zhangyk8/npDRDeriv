#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Dec 31, 2024

Description: Synthesize the outputs from repeated experiments (Simulation without positivity).
"""

import numpy as np
import pickle

#=======================================================================================#

n_lst = [500, 1000, 2000, 4000, 6000, 8000, 10000]
fac_lst = [0.75, 1, 1.5, 2, 4]
B = 1000

for n in n_lst:
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
            with open('./Results/Sim1_Nopos_theta_'+str(b)+'_n_'+str(n)+'_h'+str(fac)+'_inner_selfnorm.dat', "rb") as file:
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
            
        with open('./Syn_Results/Sim1_Nopos_theta_n_'+str(n)+'_h'+str(fac)+'_inner_selfnorm.dat', "wb") as file:
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
            with open('./Results/Sim1_Nopos_theta_'+str(b)+'_n_'+str(n)+'_h'+str(fac)+'_noselfnorm.dat', "rb") as file:
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
            
        with open('./Syn_Results/Sim1_Nopos_theta_n_'+str(n)+'_h'+str(fac)+'_noselfnorm.dat', "wb") as file:
            pickle.dump([theta_ra5_lst, theta_ra1_lst, theta_ipw5_lst, theta_ipw1_lst,
                         theta_dr5_lst, theta_sd_dr5_lst, theta_dr1_lst, theta_sd_dr1_lst,
                         theta_C_RA5_lst, theta_C_RA1_lst, theta_C_IPW5_lst, theta_C_IPW1_lst,
                         theta_C_DR5_lst, theta_C_sd_DR5_lst, theta_C_DR1_lst, theta_C_sd_DR1_lst], file)
            
            

# n = 2000

# thres_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# for thres in thres_lst:
#     # Simulation 1
#     for fac in fac_lst:
#         # Results for the derivative of dose-response curve \theta
#         theta_ra5_lst = [] 
#         theta_ra1_lst = []

#         theta_ipw5_lst = [] 
#         theta_ipw1_lst = []

#         theta_dr5_lst = [] 
#         theta_sd_dr5_lst = [] 
#         theta_dr1_lst = [] 
#         theta_sd_dr1_lst = []
        
#         theta_C_RA5_lst = [] 
#         theta_C_RA1_lst = []

#         theta_C_IPW5_lst = [] 
#         theta_C_IPW1_lst = []

#         theta_C_DR5_lst = [] 
#         theta_C_sd_DR5_lst = [] 
#         theta_C_DR1_lst = [] 
#         theta_C_sd_DR1_lst = []
#         for b in range(1, B+1):
#             with open('./Results/Sim1_Nopos_theta_'+str(b)+'_n_'+str(n)+'_h'+str(fac)+'_thres_'+str(thres)+'.dat', "rb") as file:
#                 theta_ra5, theta_ra1, theta_ipw5, theta_ipw1, theta_dr5, theta_sd5, theta_dr1, theta_sd1,\
#                 theta_C_RA5, theta_C_RA1, theta_C_IPW5, theta_C_IPW1, theta_C_DR5, theta_C_sd5,\
#                 theta_C_DR1, theta_C_sd1 = pickle.load(file)
            
#             # With positivity
#             theta_ra5_lst.append(theta_ra5)
#             theta_ra1_lst.append(theta_ra1)
            
#             theta_ipw5_lst.append(theta_ipw5)
#             theta_ipw1_lst.append(theta_ipw1)
            
#             theta_dr5_lst.append(theta_dr5)
#             theta_sd_dr5_lst.append(theta_sd5)
#             theta_dr1_lst.append(theta_dr1)
#             theta_sd_dr1_lst.append(theta_sd1)
            
#             # Without positivity
#             theta_C_RA5_lst.append(theta_C_RA5)
#             theta_C_RA1_lst.append(theta_C_RA1)
            
#             theta_C_IPW5_lst.append(theta_C_IPW5)
#             theta_C_IPW1_lst.append(theta_C_IPW1)
            
#             theta_C_DR5_lst.append(theta_C_DR5)
#             theta_C_sd_DR5_lst.append(theta_C_sd5)
#             theta_C_DR1_lst.append(theta_C_DR1)
#             theta_C_sd_DR1_lst.append(theta_C_sd1)
            
#         with open('./Syn_Results/Sim1_Nopos_theta_n_'+str(n)+'_h'+str(fac)+'_thres_'+str(thres)+'.dat', "wb") as file:
#             pickle.dump([theta_ra5_lst, theta_ra1_lst, theta_ipw5_lst, theta_ipw1_lst,
#                          theta_dr5_lst, theta_sd_dr5_lst, theta_dr1_lst, theta_sd_dr1_lst,
#                          theta_C_RA5_lst, theta_C_RA1_lst, theta_C_IPW5_lst, theta_C_IPW1_lst,
#                          theta_C_DR5_lst, theta_C_sd_DR5_lst, theta_C_DR1_lst, theta_C_sd_DR1_lst], file)