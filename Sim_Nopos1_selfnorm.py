#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Dec 22, 2024

Description: Simulation 1 with bounded treatment variation (positivity violation).
"""

import numpy as np

from npDoseResponseDR import DRCurve
from npDoseResponseDerivDR import DRDerivCurve, NeurNet, RADRDerivBC, IPWDRDerivBC, DRDRDerivBC
import sys
import pickle

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#

n = 2000

np.random.seed(job_id)
# Data generating process
X_sim = 2*np.random.rand(n) - 1
T_sim = np.sin(np.pi*X_sim) + np.random.rand(n)*0.6 - 0.3
Y_sim = T_sim**2 + T_sim**3 + 10*X_sim + np.random.normal(loc=0, scale=1, size=n)
X_dat = np.concatenate([T_sim.reshape(-1,1), X_sim.reshape(-1,1)], axis=1)

# Query points 
t_qry = np.linspace(min(T_sim)+0.2, max(T_sim)-0.2, 50)

# Compute the true conditional densities at the sample points
def CondDenTrue2(Y, X):
    reg_part = Y - np.sin(np.pi*X)
    res = ((reg_part >= -0.3) & (reg_part <= 0.3))/0.6
    return res

# Compute the true conditional densities on the sample points
true_cond = CondDenTrue2(T_sim, X_sim)


# RA estimators that assume positivity
theta_ra5 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="RA", beta_mod=NeurNet, 
                         n_iter=1000, lr=0.01, L=5, print_bw=False, delta=0.01)
theta_ra1 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="RA", beta_mod=NeurNet, 
                         n_iter=1000, lr=0.01, L=1, print_bw=False, delta=0.01)

# Bias-corrected RA estimators without positivity
theta_C_RA5 = RADRDerivBC(Y=Y_sim, X=X_dat, t_eval=t_qry, mu=NeurNet, L=5, 
                          n_iter=1000, lr=0.01, h_bar=None, kernT_bar="gaussian")
theta_C_RA1 = RADRDerivBC(Y=Y_sim, X=X_dat, t_eval=t_qry, mu=NeurNet, L=1, 
                          n_iter=1000, lr=0.01, h_bar=None, kernT_bar="gaussian")

fac_lst = [0.75, 1, 1.5, 2, 3, 4, 6, 8]
for fac in fac_lst:
    # Bandwidth choice
    h = fac*np.std(T_sim)*n**(-1/5)
    
    # IPW and DR estimators that assume positivity
    theta_ipw5 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="IPW", 
                          beta_mod=None, condTS_type='true', condTS_mod=true_cond, 
                          tau=0.001, L=5, h=h, kern="epanechnikov", h_cond=None, print_bw=True,
                          self_norm=True)
    theta_ipw1 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="IPW", 
                          beta_mod=None, condTS_type='true', condTS_mod=true_cond, 
                          tau=0.001, L=1, h=h, kern="epanechnikov", h_cond=None, print_bw=True,
                          self_norm=True)
    
    theta_dr5, theta_sd5 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="DR", 
                                    beta_mod=NeurNet, n_iter=1000, lr=0.01, 
                                    condTS_type='true', condTS_mod=true_cond, 
                                  tau=0.001, L=5, h=h, kern="epanechnikov", 
                                  h_cond=None, print_bw=False, delta=0.01, self_norm=True)
    theta_dr1, theta_sd1 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="DR", 
                                    beta_mod=NeurNet, n_iter=1000, 
                                    lr=0.01, condTS_type='true', condTS_mod=true_cond,
                                    tau=0.001, L=1, h=h, kern="epanechnikov", h_cond=None, 
                                    print_bw=False, delta=0.01, self_norm=True)
    
    # Bias-corrected IPW and DR estimators without positivity
    theta_C_IPW1, condTS1 = IPWDRDerivBC(Y=Y_sim, X=X_dat, t_eval=t_qry, L=1, h=h, 
                                         kern='epanechnikov', b=None, self_norm=True, thres_val=0.8)
    theta_C_IPW5, condTS5 = IPWDRDerivBC(Y=Y_sim, X=X_dat, t_eval=t_qry, L=5, h=h, 
                                         kern='epanechnikov', b=None, self_norm=True, thres_val=0.8)
    
    theta_C_DR1, theta_C_sd1 = DRDRDerivBC(Y=Y_sim, X=X_dat, t_eval=t_qry, mu=NeurNet, L=1, 
                                           h=h, kern='epanechnikov', n_iter=1000, lr=0.01, 
                                           b=None, thres_val=0.8, self_norm=True)
    theta_C_DR5, theta_C_sd5 = DRDRDerivBC(Y=Y_sim, X=X_dat, t_eval=t_qry, mu=NeurNet, 
                                           L=5, h=h, kern='epanechnikov', n_iter=1000, lr=0.01, 
                                           b=None, thres_val=0.8, self_norm=True)
    
    with open('./Results/Sim1_Nopos_theta_'+str(job_id)+'_h'+str(fac)+'_selfnorm.dat', "wb") as file:
        pickle.dump([theta_ra5, theta_ra1, theta_ipw5, theta_ipw1, 
                     theta_dr5, theta_sd5, theta_dr1, theta_sd1,
                     theta_C_RA5, theta_C_RA1, theta_C_IPW5, theta_C_IPW1, 
                     theta_C_DR5, theta_C_sd5, theta_C_DR1, theta_C_sd1], file)

