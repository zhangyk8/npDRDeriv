# -*- coding: utf-8 -*-

# Author: Yikun Zhang
# Last Editing: Dec 22, 2024

# Description: This script contains the implementations of the regression 
# adjustment (RA), inverse probability weighting (IPW), and doubly robust (DR) 
# estimators of the dose-response curve under the positivity condition.

import numpy as np
from rbf import KernelRetrieval
from utils1 import CondDenEst, CondDenEstKDE
from sklearn.model_selection import KFold

#=======================================================================================#

def RegAdjustDR(Y, X, t_eval, mu, L=1, multi_boot=False, B=1000):
    '''
    Estimating the dose-response curve through the regression adjustment
    (or G-computation) form.
    
    Parameters
    ----------
        Y: (n,)-array
            The outcome variables of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are the confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points.
            
        mu: scikit-learn model or any python model that can use ".fit()" and ".predict()"
            The conditional mean outcome (or regression) model of Y given X.
            
        L: int
            The number of data folds for cross-fitting. When L<= 1, no cross-fittings 
            are applied and the regression model is fitted on the entire dataset.
            (Default: L=1.)
            
        multi_boot: boolean
            An indicator of whether the multiplier bootstrap will be run. 
            (Default: multi_boot=False.)
            
        B: int
            The number of bootstrapping times. (Default: B=1000.)
            
    Return
    ----------
        m_est: (m,)-array
            The estimated dose-response curve evaluated at points "t_eval".
            
        mu_boot: (B,m)-array
            The estimated dose-response curves on bootstrapping data evaluated
            at points "t_eval". (Only return this quantity when "multi_boot=True".)
    '''
    n = X.shape[0]  ## Number of data points
    if L <= 1:
        # No cross-fittings: fit the regression model on the entire data
        mu_hat = mu.fit(X, Y)
        mu_est = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            # Define the data matrix for evaluating the fitted regression model
            X_eval = np.column_stack([t_eval[i]*np.ones(n), X[:,1:]])
            mu_est[:,i] = mu_hat.predict(X_eval)
    else:
        # Conduct L-fold cross-fittings: fit the regression model on the training fold data
        # and evaluate it on the test fold data
        kf = KFold(n_splits=L, shuffle=True, random_state=0)
        mu_est = np.zeros((n, t_eval.shape[0]))
        for tr_ind, te_ind in kf.split(X):
            X_tr = X[tr_ind,:]
            Y_tr = Y[tr_ind]
            X_te = X[te_ind,:]
            
            mu_hat = mu.fit(X_tr, Y_tr)
            for i in range(t_eval.shape[0]):
                X_eval_te = np.column_stack([t_eval[i]*np.ones(X_te.shape[0]), X_te[:,1:]])
                mu_est[te_ind,i] = mu_hat.predict(X_eval_te)
    
    if multi_boot:
        mu_boot = np.zeros((B, t_eval.shape[0]))
        for b in range(B):
            Z = np.random.randn(n, t_eval.shape[0]) + 1
            mu_boot[b,:] = np.mean(Z * mu_est, axis=0)
        m_est = np.mean(mu_est, axis=0)
        return m_est, mu_boot
    else:
        m_est = np.mean(mu_est, axis=0)
        return m_est


def IPWDR(Y, X, t_eval, condTS_type, condTS_mod, L, h, kern="epanechnikov", 
          tau=0.001, b=None, self_norm=True):
    '''
    Estimating the dose-response curve through the inverse probability weighting
    (IPW) form.
    
    Parameters
    ----------
        Y: (n,)-array
            The outcome variables of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are the confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points.
            
        condTS_type: str
            Specifying the model type for estimating the conditional density of
            the treatment variable T given the covariate vector S.
            
        condTS_mod: scikit-learn model or any python model that can use ".fit()" and ".predict()"
            The regression model for estimating the conditional density of T given S.
            
        L: int
            The number of data folds for cross-fitting. When L<= 1, no cross-fittings 
            are applied and the regression model is fitted on the entire dataset.
            (Default: L=1.)
            
        h: float
            The bandwidth parameter.
            
        kern: str
            The name of the kernel function. (Default: kern="epanechnikov".)
            
        tau: float
            The threshold value that lower bounds the estimated conditional density
            values. (Default: tau=0.001.)
            
        b: float
            The bandwidth parameter for the kernel-smoothed conditional density
            estimation methods. (Default: b=None.)
            
        self_norm: boolean
            An indicator of whether the self-normalized version is implemented.
            (Default: self_norm=True.)
            
    Return
    ----------
        m_est: (m,)-array
            The estimated dose-response curve evaluated at points "t_eval".
            
        cond_est_full: (n,)-array
            The estimated conditional density function of T given S evaluated at
            the n observed data points.
    '''
    
    kern_type = kern
    kern, sigmaK_sq, K_sq = KernelRetrieval(kern_type)
    n = X.shape[0]  ## Number of data points
    if L <= 1:
        # No cross-fittings: fit the conditional density model on the entire data
        if condTS_type == 'true':
            condTS_est = condTS_mod
        elif condTS_type == 'kde':
            condTS_est = CondDenEstKDE(X[:,0], X[:,1:], reg_mod=condTS_mod, y_eval=X[:,0], 
                                       x_eval=X[:,1:], kern=kern_type, b=b)
        else:
            condTS_est = CondDenEst(X[:,0], X[:,1:], reg_mod=condTS_mod, y_eval=X[:,0], 
                                    x_eval=X[:,1:], kern='gaussian', b=b)
        condTS_est[condTS_est < tau] = tau
        
        m_hat = np.zeros((n, t_eval.shape[0]))
        norm_w = np.zeros((t_eval.shape[0],))
        for i in range(t_eval.shape[0]):
            # Self-normalizing weights
            norm_w[i] = np.sum(kern((t_eval[i] - X[:,0])/h) / condTS_est) / h
            m_hat[:,i] = kern((t_eval[i] - X[:,0])/h) * Y / (h * condTS_est)

        if self_norm:
            # Self-normalized IPW estimator
            m_hat = m_hat / norm_w
            m_est = np.sum(m_hat, axis=0, where=~np.isnan(m_hat))
        else:
            m_est = np.mean(m_hat, axis=0, where=~np.isnan(m_hat))
        cond_est_full = condTS_est.copy()
    else:
        # Conduct L-fold cross-fittings: fit the conditional density model on the training fold 
        # data and evaluate it on the test fold data
        kf = KFold(n_splits=L, shuffle=True, random_state=0)
        m_hat = np.zeros((n, t_eval.shape[0]))
        norm_w = np.zeros((t_eval.shape[0],))
        cond_est_full = np.zeros((n,))
        for tr_ind, te_ind in kf.split(X):
            X_tr = X[tr_ind,:]
            X_te = X[te_ind,:]
            Y_te = Y[te_ind]
            
            if condTS_type == 'true':
                condTS_est = condTS_mod[te_ind]
            elif condTS_type == 'kde':
                condTS_est = CondDenEstKDE(X_tr[:,0], X_tr[:,1:], reg_mod=condTS_mod, 
                                           y_eval=X_te[:,0], x_eval=X_te[:,1:], kern=kern_type, b=b)
            else:
                condTS_est = CondDenEst(X_tr[:,0], X_tr[:,1:], reg_mod=condTS_mod, 
                                        y_eval=X_te[:,0], x_eval=X_te[:,1:], kern='gaussian', b=b)
            condTS_est[condTS_est < tau] = tau
            cond_est_full[te_ind] = condTS_est
            for i in range(t_eval.shape[0]):
                # Self-normalizing weights
                w = np.sum(kern((t_eval[i] - X[te_ind,0])/h) / condTS_est) / h
                norm_w[i] = norm_w[i] + w
                m_hat[te_ind,i] = kern((t_eval[i] - X[te_ind,0])/h) * Y_te / (h * condTS_est)

        if self_norm:
            norm_w[norm_w == 0] = 1
            m_est = np.sum(m_hat, axis=0, where=~np.isnan(m_hat)) / norm_w
        else:
            m_est = np.mean(m_hat, axis=0, where=~np.isnan(m_hat))
    return m_est, cond_est_full



def DRDR(Y, X, t_eval, mu, condTS_type, condTS_mod, L, h, kern="epanechnikov", 
         tau=0.001, b=None, self_norm=True):
    '''
    Estimating the dose-response curve through the doubly robust (DR) form.
    
    Parameters
    ----------
        Y: (n,)-array
            The outcome variables of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are the confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points.
            
        mu: scikit-learn model or any python model that can use ".fit()" and ".predict()"
            The conditional mean outcome (or regression) model of Y given X.
            
        condTS_type: str
            Specifying the model type for estimating the conditional density of
            the treatment variable T given the covariate vector S.
            
        condTS_mod: scikit-learn model or any python model that can use ".fit()" and ".predict()"
            The regression model for estimating the conditional density of T given S.
            
        L: int
            The number of data folds for cross-fitting. When L<= 1, no cross-fittings 
            are applied and the regression model is fitted on the entire dataset.
            
        h: float
            The bandwidth parameter.
            
        kern: str
            The name of the kernel function. (Default: kern="epanechnikov".)
            
        tau: float
            The threshold value that lower bounds the estimated conditional density
            values. (Default: tau=0.001.)
            
        b: float
            The bandwidth parameter for the kernel-smoothed conditional density
            estimation methods. (Default: b=None.)
            
        self_norm: boolean
            An indicator of whether the self-normalized version is implemented.
            (Default: self_norm=True.)
            
    Return
    ----------
        m_est: (m,)-array
            The estimated dose-response curve evaluated at points "t_eval".
            
        sd_est: (m,)-array
            The estimated asymptotic stdndard deviation of the DR estimator 
            evaluated at points "t_eval".
    '''
    
    kern_type = kern
    kern, sigmaK_sq, K_sq = KernelRetrieval(kern)
    n = X.shape[0]  ## Number of data points
    if L <= 1:
        # No cross-fittings: fit the conditional density model and the regression model on the entire data
        if condTS_type == 'true':
            condTS_est = condTS_mod
        elif condTS_type == 'kde':
            condTS_est = CondDenEstKDE(X[:,0], X[:,1:], reg_mod=condTS_mod, 
                                       y_eval=X[:,0], x_eval=X[:,1:], kern=kern_type, b=b)
        else:
            condTS_est = CondDenEst(X[:,0], X[:,1:], reg_mod=condTS_mod, y_eval=X[:,0], 
                                    x_eval=X[:,1:], kern='gaussian', b=b)
        condTS_est[condTS_est < tau] = tau
        mu_fit = mu.fit(X, Y)
        mu_hat = np.zeros((n, t_eval.shape[0]))
        IPW_hat = np.zeros((n, t_eval.shape[0]))
        norm_w = np.zeros((t_eval.shape[0],))
        for i in range(t_eval.shape[0]):
            # Define the data matrix for evaluating the fitted regression model
            X_eval = np.column_stack([t_eval[i]*np.ones(n), X[:,1:]])
            mu_hat[:,i] = mu_fit.predict(X_eval)
            IPW_hat[:,i] = kern((t_eval[i] - X[:,0])/h) * (Y - mu_hat[:,i]) / (h * condTS_est)
            # Self-normalizing weights
            norm_w[i] = np.sum(kern((t_eval[i] - X[:,0])/h) / condTS_est) / (n * h)
            
        if self_norm:
            IPW_hat = IPW_hat / norm_w
        # Add up the IPW and RA components
        m_hat = IPW_hat + mu_hat
        m_est = np.mean(m_hat, axis=0, where=~np.isnan(m_hat))
        
        # Estimate the variance of m(t) using the square of the influence function
        var_est = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            var_est[:,i] = (IPW_hat[:,i] + (mu_hat[:,i] - m_est[i]))**2 * h
        sd_est = np.sqrt(np.mean(var_est, axis=0)/(n*h))
    else:
        # Conduct L-fold cross-fittings: fit the reciprocal of the conditional model 
        # and the regression model on the training fold data and evaluate it on the test fold data
        kf = KFold(n_splits=L, shuffle=True, random_state=0)
        mu_hat = np.zeros((n, t_eval.shape[0]))
        IPW_hat = np.zeros((n, t_eval.shape[0]))
        norm_w = np.zeros((t_eval.shape[0],))
        cond_est_full = np.zeros((n,))
        for tr_ind, te_ind in kf.split(X):
            X_tr = X[tr_ind,:]
            Y_tr = Y[tr_ind]
            X_te = X[te_ind,:]
            Y_te = Y[te_ind]
            
            if condTS_type == 'true':
                condTS_est = condTS_mod[te_ind]
            elif condTS_type == 'kde':
                condTS_est = CondDenEstKDE(X_tr[:,0], X_tr[:,1:], reg_mod=condTS_mod, 
                                           y_eval=X_te[:,0], x_eval=X_te[:,1:], kern=kern_type, b=b)
            else:
                condTS_est = CondDenEst(X_tr[:,0], X_tr[:,1:], reg_mod=condTS_mod, 
                                        y_eval=X_te[:,0], x_eval=X_te[:,1:], kern='gaussian', b=b)
            condTS_est[condTS_est < tau] = tau
            cond_est_full[te_ind] = condTS_est
            mu_fit = mu.fit(X_tr, Y_tr)
            for i in range(t_eval.shape[0]):
                X_eval_te = np.column_stack([t_eval[i]*np.ones(X_te.shape[0]), X_te[:,1:]])
                mu_hat[te_ind,i] = mu_fit.predict(X_eval_te)
                IPW_hat[te_ind,i] = kern((t_eval[i] - X[te_ind,0])/h) * (Y_te - mu_hat[te_ind,i]) / (h * condTS_est)
                
                # Self-normalizing weights
                w = np.sum(kern((t_eval[i] - X[te_ind,0])/h) / condTS_est) / (n * h)
                norm_w[i] = norm_w[i] + w
        
        if self_norm:
            IPW_hat = IPW_hat / norm_w
        # Add up the IPW and RA components
        m_hat = IPW_hat + mu_hat
        m_est = np.mean(m_hat, axis=0, where=~np.isnan(m_hat))
        
        # Estimate the variance of m(t) using the square of the influence function
        var_est = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            var_est[:,i] = (IPW_hat[:,i] + (mu_hat[:,i] - m_est[i]))**2 * h
        sd_est = np.sqrt(np.mean(var_est, axis=0)/(n*h))
    return m_est, sd_est


def DRCurve(Y, X, t_eval=None, est="RA", mu=None, condTS_type=None, condTS_mod=None, 
            L=1, h=None, kern="epanechnikov", tau=0.001, h_cond=None, self_norm=True, 
            print_bw=True):
    '''
    Dose-response curve estimation under the positivity condition.
    
    Parameters
    ----------
        Y: (n,)-array
            The outcome variables of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points. (Default: t_eval=None. 
            Then, t_eval=X[:,0], which consists of the observed treatment variables.)
            
        est: str
            The type of the dose-response curve estimator. (Default: est="RA". 
            Other choices include "IPW" and "DR".)
            
        mu: scikit-learn model or any python model that can use ".fit()" and ".predict()"
            The conditional mean outcome (or regression) model of Y given X.
            
        condTS_type: str
            Specifying the model type for estimating the conditional density of
            the treatment variable T given the covariate vector S.
            
        condTS_mod: scikit-learn model or any python model that can use ".fit()" and ".predict()"
            The regression model for estimating the conditional density of T given S.
            
        L: int
            The number of data folds for cross-fitting. When L<= 1, no cross-fittings 
            are applied and the regression model is fitted on the entire dataset.
            
        h: float
            The bandwidth parameter for the IPW/DR estimator. (Default: h=None. 
            Then the Silverman's rule of thumb is applied; see Chen et al.(2016) 
            for details.)
            
        kern: str
            The name of the kernel function. (Default: kern="epanechnikov".)
            
        tau: float
            The threshold value that lower bounds the estimated conditional density
            values. (Default: tau=0.001.)
            
        h_cond: float
            The bandwidth parameter for the kernel-smoothed conditional density
            estimation methods. (Default: b=None.)
            
        self_norm: boolean
            An indicator of whether the self-normalized version is implemented.
            (Default: self_norm=True.)
            
        print_bw: boolean
            The indicator of whether the current bandwidth parameters should be
            printed to the console. (Default: print_bw=True.)
    
    Return
    ----------
        m_est: (m,)-array
            The estimated dose-response curve evaluated at points "t_eval".
            
        sd_est: (m,)-array (if est="DR")
            The estimated asymptotic standard deviation of the DR estimator 
            evaluated at points "t_eval".
    '''
    
    if t_eval is None: 
        t_eval = X[:,0].copy()
    
    n = X.shape[0]  ## Number of data points
    
    if h is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        h = (4/3)**(1/5)*(n**(-1/5))*np.std(X[:,0])
    
    if print_bw:
        print("The current bandwidth for the "+str(est)+" estimator is "+ str(h) + ".\n")
    
    if est == "RA":
        m_est = RegAdjustDR(Y, X, t_eval, mu, L)
    elif est == "IPW":
        m_est, cond_est = IPWDR(Y, X, t_eval, condTS_type, condTS_mod, L, h, kern, tau, h_cond, 
                                self_norm)
    else:
        m_est = DRDR(Y, X, t_eval, mu, condTS_type, condTS_mod, L, h, kern, tau, h_cond, self_norm)
    
    return m_est

