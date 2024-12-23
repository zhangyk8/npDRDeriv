#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Dec 18, 2024

Description: Case study on the job corps program in the US (Final).
"""

import numpy as np
import pandas as pd
import sklearn

from npDoseResponseDR import DRCurve
from npDoseResponseDerivDR import DRDerivCurve, NeurNet
import pickle

import Supplement

#=======================================================================================#

# Read and preprocess the data as in Colangelo and Lee 2020
data = pd.read_csv('./job_corps.csv', index_col=0)
data = data.sample(frac=1, random_state=20)

data = pd.concat([data.select_dtypes(exclude='int64'),
                  pd.get_dummies(data.select_dtypes('int64').astype('category'),
                                 drop_first=True).astype('int64')],
                 axis=1)
X = data.drop(['d','y'], axis=1).values #define covariate vector, excluding T and Y
T = data['d'].values # define treatment vector
Y = data['y'].values # define outcome vector

# Evaluation points
t_list_new = np.arange(40, 4001, 40)
 
# Standardize the data as in Colangelo and Lee 2020
dat = np.column_stack([Y,T,X])
dat = sklearn.preprocessing.StandardScaler().fit_transform(pd.DataFrame(dat))
Y_new = dat[:,0]
X_dat = dat[:,1:]
t_qry = (t_list_new - np.mean(T))/np.std(T)

# Use the same conditional density model as in Colangelo and Lee 2020
model_nn2 = Supplement.NeuralNet2_emp_app(k=138,
                                          lr=0.05, 
                                          momentum=0.3,
                                          epochs=100,
                                          weight_decay=0.15)

model_nn1 = Supplement.NeuralNet1_emp_app(k=139,
                                          lr=0.15,
                                          momentum=0.9,
                                          epochs=100,
                                          weight_decay=0.05)

np.random.seed(123)

h1 = 223/np.std(T)

m_est_dr1, sd_est_dr1 = DRCurve(Y=Y_new, X=X_dat, t_eval=t_qry, est="DR", 
                            mu=model_nn1, condTS_type='reg', condTS_mod=model_nn2,
                            tau=0.001, L=1, h=h1, kern="epanechnikov", h_cond=None,
                            print_bw=False)

m_est_dr5, sd_est_dr5 = DRCurve(Y=Y_new, X=X_dat, t_eval=t_qry, est="DR", 
                            mu=model_nn1, condTS_type='reg', condTS_mod=model_nn2,
                            tau=0.001, L=5, h=h1, kern="epanechnikov", h_cond=None,
                            print_bw=False)

theta_dr5, theta_sd5 = DRDerivCurve(Y=Y_new, X=X_dat, t_eval=t_qry, est="DR", 
                                beta_mod=NeurNet, n_iter=1000, 
                                lr=0.1, condTS_type='reg', condTS_mod=model_nn2, 
                                tau=0.1, L=5, h=h1, kern="epanechnikov", 
                                h_cond=None, print_bw=True, delta=0.01)

theta_dr1, theta_sd1 = DRDerivCurve(Y=Y_new, X=X_dat, t_eval=t_qry, est="DR", 
                                beta_mod=NeurNet, n_iter=1000, 
                                lr=0.1, condTS_type='reg', condTS_mod=model_nn2, 
                                tau=0.001, L=1, h=h1, kern="epanechnikov", 
                                h_cond=None, print_bw=True, delta=0.01)

m_est_ra5 = DRCurve(Y=Y_new, X=X_dat, t_eval=t_qry, est="RA", mu=model_nn1, 
                    L=5, h=None, kern="epanechnikov", print_bw=False)
m_est_ra1 = DRCurve(Y=Y_new, X=X_dat, t_eval=t_qry, est="RA", mu=model_nn1, 
                    L=1, h=None, kern="epanechnikov", print_bw=False)

theta_ra5 = DRDerivCurve(Y=Y_new, X=X_dat, t_eval=t_qry, est="RA", beta_mod=NeurNet, 
                n_iter=1000, lr=0.1, L=5, print_bw=False, delta=0.01)

theta_ra1 = DRDerivCurve(Y=Y_new, X=X_dat, t_eval=t_qry, est="RA", beta_mod=NeurNet, 
                n_iter=1000, lr=0.1, L=1, print_bw=False, delta=0.01)


# m_kde_dr1, sd_kde_dr1 = DRCurve(Y=Y_new, X=X_dat, t_eval=t_qry, est="DR", 
#                             mu=model_nn1, condTS_type='kde', condTS_mod=model_nn2,
#                             tau=0.001, L=1, h=h1, kern="epanechnikov", h_cond=None,
#                             print_bw=False, bnd_cor=False)

# m_kde_dr5, sd_kde_dr5 = DRCurve(Y=Y_new, X=X_dat, t_eval=t_qry, est="DR", 
#                             mu=model_nn1, condTS_type='kde', condTS_mod=model_nn2,
#                             tau=0.001, L=5, h=h1, kern="epanechnikov", h_cond=None,
#                             print_bw=False, bnd_cor=False)

# theta_kde_dr5, theta_kde_sd5 = DRDerivCurve(Y=Y_new, X=X_dat, t_eval=t_qry, est="DR", 
#                                 beta_mod=NeurNet, n_iter=1000, 
#                                 lr=0.1, condTS_type='kde', condTS_mod=model_nn2, 
#                                 tau=0.001, L=5, h=h1, kern="epanechnikov", 
#                                 h_cond=None, print_bw=True, bnd_cor=False, delta=0.01)

# theta_kde_dr1, theta_kde_sd1 = DRDerivCurve(Y=Y_new, X=X_dat, t_eval=t_qry, est="DR", 
#                                 beta_mod=NeurNet, n_iter=1000, 
#                                 lr=0.1, condTS_type='kde', condTS_mod=model_nn2, 
#                                 tau=0.001, L=1, h=h1, kern="epanechnikov", 
#                                 h_cond=None, print_bw=True, bnd_cor=False, delta=0.01)

    
with open('./Syn_Results/Job_Corps_est_lr_'+str(0.1)+'_final1.dat', "wb") as file:
    pickle.dump([m_est_ra5, m_est_ra1, theta_ra5, theta_ra1, m_est_dr5, sd_est_dr5,\
                 m_est_dr1, sd_est_dr1, theta_dr5, theta_sd5, theta_dr1, theta_sd1], file)
    