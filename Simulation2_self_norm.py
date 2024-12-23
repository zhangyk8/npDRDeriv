#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Dec 11, 2024

Description: Simulation 2 with non-separable error in the outcome model.
"""

import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from npDoseResponseDR import DRCurve
from npDoseResponseDerivDR import DRDerivCurve, NeurNet
import sys
import pickle

import Supplement

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#

# Compute the true conditional densities at the sample points
def CondDenTrue(Y, X):
    res = np.exp(-(Y - scipy.stats.norm.cdf(3*np.dot(X, theta)) + 1/2)**2/(2*0.75**2))/(0.75*np.sqrt(2*np.pi))
    return res


reg_mod_lst = ['LR', 'RF', 'NN']
cond_type = ['true', 'kde', 'reg']
fac_lst = [0.75, 1, 1.25, 1.5, 2, 4]
n_lst = [1000, 2000]


for reg in reg_mod_lst:
    for fac in fac_lst:
        for cond in cond_type:
            for n in n_lst:
                rho = 0.5  # correlation between adjacent Xs
                if n == 10000:
                    d = 100   # Dimension of the confounding variables
                    model_nn2_n10000 = Supplement.NeuralNet2_n10000(k=d, 
                                                          lr=0.4,
                                                          momentum = 0.0, 
                                                          epochs=100,
                                                          weight_decay=0.075)
                    cond_reg_mod = model_nn2_n10000
                else:
                    d = 20   # Dimension of the confounding variables
                    model_nn2_n1000 = Supplement.NeuralNet2_n1000(k=d, 
                                                  lr=0.01,
                                                  momentum = 0.9, 
                                                  epochs=100,
                                                  weight_decay=0.3)
                    cond_reg_mod = model_nn2_n1000
                

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
                
                
                if reg == 'LR':
                    reg_mod = LinearRegression()
                elif reg == 'RF':
                    reg_mod = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=1)
                elif reg == 'NN':
                    reg_mod = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', 
                                           learning_rate='adaptive', learning_rate_init=0.1, 
                                           random_state=1, max_iter=200)
                # Bandwidth choice
                h = fac*np.std(T_sim)*n**(-1/5)
                
                t_qry = np.linspace(-2, 2, 81)
                m_est_ra5 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="RA", mu=reg_mod, 
                                    L=5, h=None, kern="epanechnikov", print_bw=False)

                m_est_ra1 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="RA", mu=reg_mod, 
                                    L=1, h=None, kern="epanechnikov", print_bw=False)
                
                if cond == 'true':
                    # True conditional density values
                    true_cond = CondDenTrue(T_sim, X_sim)
                    cond_mod = true_cond
                elif cond == 'kde':
                    # Conditional density model
                    regr_nn2 = MLPRegressor(hidden_layer_sizes=(20,), activation='relu', learning_rate='adaptive', 
                                    learning_rate_init=0.1, random_state=1, max_iter=200)
                    cond_mod = regr_nn2
                else:
                    cond_mod = cond_reg_mod
                    
                m_est_ipw5 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="IPW", mu=None, 
                                          condTS_type=cond, condTS_mod=cond_mod, 
                                          tau=0.001, L=5, h=h, kern="epanechnikov", h_cond=None, 
                                          print_bw=True, self_norm=True)

                m_est_ipw1 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="IPW", mu=None, 
                                          condTS_type=cond, condTS_mod=cond_mod, tau=0.001, 
                                          L=1, h=h, kern="epanechnikov", h_cond=None, print_bw=False,
                                          self_norm=True)
                
                m_est_dr5, sd_est_dr5 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="DR", 
                                                mu=reg_mod, condTS_type=cond, condTS_mod=cond_mod,
                                                tau=0.001, L=5, h=h, kern="epanechnikov", h_cond=None,
                                                print_bw=False, self_norm=True)

                m_est_dr1, sd_est_dr1 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="DR", 
                                                mu=reg_mod, condTS_type=cond, condTS_mod=cond_mod, 
                                                tau=0.001, L=1, h=h, kern="epanechnikov", h_cond=None, 
                                                print_bw=False, self_norm=True)
                
                with open('./Results/Simulation2_m_est'+str(job_id)+'_'+str(reg)+'_h'+str(fac)+'_condmod_'+str(cond)+'_n_'+str(n)+'_selfnorm.dat', "wb") as file:
                    pickle.dump([m_est_ra5, m_est_ra1, m_est_ipw5, m_est_ipw1, 
                                 m_est_dr5, sd_est_dr5, m_est_dr1, sd_est_dr1], file)
                
                if reg == 'LR':
                    reg_mod = LinearRegression()
                elif reg == 'RF':
                    reg_mod = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=1)
                elif reg == 'NN':
                    reg_mod = NeurNet
                
                theta_ra5 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="RA", beta_mod=reg_mod, 
                                n_iter=1000, lr=0.01, L=5, print_bw=False, delta=0.01)
                theta_ra1 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="RA", beta_mod=reg_mod, 
                                n_iter=1000, lr=0.01, L=1, print_bw=False, delta=0.01)
                
                theta_ipw5 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="IPW", 
                                          beta_mod=None, condTS_type=cond, condTS_mod=cond_mod, 
                                          tau=0.001, L=5, h=h, kern="epanechnikov", h_cond=None, 
                                          print_bw=True, self_norm=True)
                theta_ipw1 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="IPW", 
                                          beta_mod=None, condTS_type=cond, condTS_mod=cond_mod, 
                                          tau=0.001, L=1, h=h, kern="epanechnikov", h_cond=None, 
                                          print_bw=True, self_norm=True)
                
                theta_dr5, theta_sd5 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="DR", 
                                                            beta_mod=reg_mod, n_iter=1000, 
                                                  lr=0.01, condTS_type=cond, condTS_mod=cond_mod, 
                                                  tau=0.001, L=5, h=h, kern="epanechnikov", 
                                                  h_cond=None, print_bw=True, delta=0.01, self_norm=True)
                theta_dr1, theta_sd1 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="DR", 
                                                            beta_mod=reg_mod, n_iter=1000, 
                                                  lr=0.01, condTS_type=cond, condTS_mod=cond_mod, 
                                                  tau=0.001, L=1, h=h, kern="epanechnikov", h_cond=None, 
                                                  print_bw=True, delta=0.01, self_norm=True)
                
                
                with open('./Results/Simulation2_theta_est'+str(job_id)+'_'+str(reg)+'_h'+str(fac)+'_condmod_'+str(cond)+'_n_'+str(n)+'_selfnorm.dat', "wb") as file:
                    pickle.dump([theta_ra5, theta_ra1, theta_ipw5, theta_ipw1, 
                                 theta_dr5, theta_sd5, theta_dr1, theta_sd1], file)
        

