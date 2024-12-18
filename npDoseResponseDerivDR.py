# -*- coding: utf-8 -*-

# Author: Yikun Zhang
# Last Editing: Nov 12, 2024

# Description: This script contains the implementations of the IPW and doubly 
# robust estimators of the derivative of a dose-response curve under the 
# positivity condition.

import numpy as np
from rbf import KernelRetrieval
from utils1 import BndKern, CondDenEst, CondDenEstKDE
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator

import torch
import torch.nn as nn
import torch.optim as optim

#=======================================================================================#


## Define the neural network
class NeurNet(nn.Module):
    def __init__(self, input_size):
        super(NeurNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)  # First layer
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(100, 50) # Second layer
        self.fc3 = nn.Linear(50, 1)
        
        # Apply Kaiming initialization to each layer
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='linear')
        
        self.double()

    def forward(self, x):
        x = self.silu(self.fc1(x))
        x = self.silu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def train(mod, X_train, Y_train, lr=0.1, n_epochs=10):
    # Initialize the model, loss function, and optimizer
    model = mod(input_size=X_train.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.7)
    
    for epoch in range(n_epochs):
        model.train()
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()    # Zero the gradients
        loss.backward()          # Backpropagate
        optimizer.step()         # Update weights
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')
        
    return model


def RADRDeriv(Y, X, t_eval, mu, L=1, n_iter=1000, lr=0.1, multi_boot=False, B=1000):
    n = X.shape[0]  ## Number of data points
    if L <= 1:
        # No cross-fittings: fit the regression model on the entire data
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y.reshape(-1,1))
        NN_fit = train(mu, X_tensor, Y_tensor, lr=lr, n_epochs=n_iter)
        beta_est = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            # Define the data matrix for evaluating the fitted regression model
            X_eval = np.column_stack([t_eval[i]*np.ones(n), X[:,1:]])
            X_eval_tensor = torch.from_numpy(X_eval)
            for j in range(X_eval.shape[0]):
                # Compute the gradient of the fitted regression model with respect to the first coordinate
                x = X_eval_tensor[j,:]
                x = x.clone().detach().requires_grad_(True)
                y = NN_fit(x)
                y_scalar = y[0]
                y_scalar.backward()
                x_grad = x.grad
                beta_est[j,i] = x_grad[0].item()
            
        # theta_est = np.mean(beta_est, axis=0)
    else:
        # Conduct L-fold cross-fittings: fit the regression model on the training fold data
        # and evaluate it on the test fold data
        kf = KFold(n_splits=L, shuffle=True, random_state=0)
        beta_est = np.zeros((n, t_eval.shape[0]))
        for tr_ind, te_ind in kf.split(X):
            X_tr = X[tr_ind,:]
            Y_tr = Y[tr_ind]
            X_te = X[te_ind,:]
            
            X_tr_tensor = torch.from_numpy(X_tr)
            Y_tr_tensor = torch.from_numpy(Y_tr.reshape(-1,1))
            NN_fit = train(mu, X_tr_tensor, Y_tr_tensor, lr=lr, n_epochs=n_iter)
            for i in range(t_eval.shape[0]):
                # Define the data matrix for evaluating the fitted regression model
                X_eval_te = np.column_stack([t_eval[i]*np.ones(X_te.shape[0]), X_te[:,1:]])
                X_eval_te_tensor = torch.from_numpy(X_eval_te)
                beta_hat = np.zeros((X_eval_te.shape[0],))
                for j in range(X_eval_te.shape[0]):
                    # Compute the gradient of the fitted regression model with respect to the first coordinate
                    x = X_eval_te_tensor[j,:]
                    x = x.clone().detach().requires_grad_(True)
                    y = NN_fit(x)
                    y_scalar = y[0]
                    y_scalar.backward()
                    x_grad = x.grad
                    beta_hat[j] = x_grad[0].item()
                beta_est[te_ind,i] = beta_hat
                    
        # theta_est = np.mean(beta_est, axis=0)
    if multi_boot:
        theta_boot = np.zeros((B, t_eval.shape[0]))
        for b in range(B):
            Z = np.random.randn(n, t_eval.shape[0]) + 1
            theta_boot[b,:] = np.mean(Z * beta_est, axis=0)
        theta_est = np.mean(beta_est, axis=0)
        return theta_est, theta_boot
    else:
        theta_est = np.mean(beta_est, axis=0)
        return theta_est


def RADRDerivSKLearn(Y, X, t_eval, mu, L=1, delta=0.01):
    n = X.shape[0]  ## Number of data points
    if L <= 1:
        # No cross-fittings: fit the regression model on the entire data
        mu_fit = mu.fit(X, Y)
        t_new = np.linspace(np.min(t_eval)-delta, np.max(t_eval)+delta, t_eval.shape[0]+1)
        beta_est = np.zeros((n, t_new.shape[0]))
        for i in range(t_new.shape[0]):
            # Define the data matrix for evaluating the fitted regression model
            X_eval = np.column_stack([t_new[i]*np.ones(n), X[:,1:]])
            beta_est[:,i] = mu_fit.predict(X_eval)
            
        theta_est = np.diff(np.mean(beta_est, axis=0))/np.diff(t_new)
    else:
        # Conduct L-fold cross-fittings: fit the regression model on the training fold data
        # and evaluate it on the test fold data
        kf = KFold(n_splits=L, shuffle=True, random_state=0)
        t_new = np.linspace(np.min(t_eval)-delta, np.max(t_eval)+delta, t_eval.shape[0]+1)
        beta_est = np.zeros((n, t_new.shape[0]))
        for tr_ind, te_ind in kf.split(X):
            X_tr = X[tr_ind,:]
            Y_tr = Y[tr_ind]
            X_te = X[te_ind,:]
            
            mu_fit = mu.fit(X_tr, Y_tr)
            for i in range(t_new.shape[0]):
                # Define the data matrix for evaluating the fitted regression model
                X_eval_te = np.column_stack([t_new[i]*np.ones(X_te.shape[0]), X_te[:,1:]])
                beta_est[te_ind,i] = mu_fit.predict(X_eval_te)
                    
        theta_est = np.diff(np.mean(beta_est, axis=0))/np.diff(t_new)
    return theta_est


def IPWDRDeriv(Y, X, t_eval, condTS_type, condTS_mod, L, h, kern, tau=0.01, b=None,
               self_norm=True, bnd_cor=True):
    kern_type = kern
    kern, sigmaK_sq, K_sq = KernelRetrieval(kern)
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
        beta_hat = np.zeros((n, t_eval.shape[0]))
        norm_w = np.zeros((t_eval.shape[0],))
        for i in range(t_eval.shape[0]):
            if bnd_cor:
                if t_eval[i] < np.min(X[:,0]) or t_eval[i] > np.max(X[:,0]):
                    beta_hat[:,i] = 0
                    norm_w[i] = 1
                elif t_eval[i] < np.min(X[:,0]) + h:
                    alpha = (t_eval[i] - np.min(X[:,0]))/h
                    bndkern = BndKern((X[:,0] - t_eval[i])/h, kern=kern, deriv_ord=1, alpha=alpha, bnd='left')
                    # Self-normalizing weights
                    norm_w[i] = np.sum(bndkern / condTS_est) / h
                    beta_hat[:,i] = ((X[:,0] - t_eval[i])/h) * bndkern * Y / (h**2 * condTS_est)
                elif t_eval[i] > np.max(X[:,0]) - h:
                    alpha = (np.max(X[:,0]) - t_eval[i])/h
                    bndkern = BndKern((X[:,0] - t_eval[i])/h, kern=kern, deriv_ord=1, alpha=alpha, bnd='right')
                    # Self-normalizing weights
                    norm_w[i] = np.sum(bndkern / condTS_est) / h
                    beta_hat[:,i] = ((X[:,0] - t_eval[i])/h) * bndkern * Y / (h**2 * condTS_est)
                else:
                    # Self-normalizing weights
                    norm_w[i] = np.sum(kern((t_eval[i] - X[:,0])/h) / condTS_est) / h
                    beta_hat[:,i] = ((X[:,0] - t_eval[i])/h) * kern((t_eval[i] - X[:,0])/h) * Y / (h**2 * sigmaK_sq * condTS_est)
            else:
                # Self-normalizing weights
                norm_w[i] = np.sum(kern((t_eval[i] - X[:,0])/h) / condTS_est) / h
                beta_hat[:,i] = ((X[:,0] - t_eval[i])/h) * kern((t_eval[i] - X[:,0])/h) * Y / (h**2 * sigmaK_sq * condTS_est)

        if self_norm:
            theta_est = np.sum(beta_hat, axis=0) / norm_w
        else:
            theta_est = np.mean(beta_hat, axis=0)
    else:
        # Conduct L-fold cross-fittings: fit the conditional density model on the training fold 
        # data and evaluate it on the test fold data
        kf = KFold(n_splits=L, shuffle=True, random_state=0)
        beta_hat = np.zeros((n, t_eval.shape[0]))
        norm_w = np.zeros((t_eval.shape[0],))
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
            for i in range(t_eval.shape[0]):
                if bnd_cor:
                    if t_eval[i] < np.min(X[te_ind,0]) or t_eval[i] > np.max(X[te_ind,0]):
                        beta_hat[te_ind,i] = 0
                    elif t_eval[i] < np.min(X[te_ind,0]) + h:
                        alpha = (t_eval[i] - np.min(X[te_ind,0]))/h
                        bndkern = BndKern((X[te_ind,0] - t_eval[i])/h, kern=kern, deriv_ord=1, 
                                          alpha=alpha, bnd='left')
                        # Self-normalizing weights
                        w = np.sum(bndkern / condTS_est) / h
                        if ~np.isnan(w) and w != np.inf:
                            norm_w[i] = norm_w[i] + w
                            beta_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * bndkern * Y_te / (condTS_est * h**2)
                        else:
                            beta_hat[te_ind,i] = 0
                    elif t_eval[i] > np.max(X[te_ind,0]) - h:
                        alpha = (np.max(X[te_ind,0]) - t_eval[i])/h
                        bndkern = BndKern((X[te_ind,0] - t_eval[i])/h, kern=kern, deriv_ord=1, 
                                          alpha=alpha, bnd='right')
                        # Self-normalizing weights
                        w = np.sum(bndkern / condTS_est) / h
                        if ~np.isnan(w) and w != np.inf:
                            norm_w[i] = norm_w[i] + w
                            beta_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * bndkern * Y_te / (condTS_est * h**2)
                        else:
                            beta_hat[te_ind,i] = 0
                    else:
                        w = np.sum(kern((t_eval[i] - X[te_ind,0])/h) / condTS_est) / h
                        if ~np.isnan(w) and w != np.inf:
                            norm_w[i] = norm_w[i] + w
                        beta_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * kern((t_eval[i] - X[te_ind,0])/h) * Y_te / (condTS_est * sigmaK_sq * h**2)
                else:
                    # Self-normalizing weights
                    w = np.sum(kern((t_eval[i] - X[te_ind,0])/h) / condTS_est) / h
                    if ~np.isnan(w) and w != np.inf:
                        norm_w[i] = norm_w[i] + w
                    beta_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * kern((t_eval[i] - X[te_ind,0])/h) * Y_te / (h**2 * sigmaK_sq * condTS_est)

        if self_norm:
            norm_w[norm_w == 0] = 1
            theta_est = np.sum(beta_hat, axis=0) / norm_w
        else:
            theta_est = np.mean(beta_hat, axis=0)
    return theta_est


def DRDRDeriv(Y, X, t_eval, mu, condTS_type, condTS_mod, L, h, kern, n_iter=1000, lr=0.1, 
              tau=0.01, b=None, bnd_cor=True):
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
            condTS_est = CondDenEst(X[:,0], X[:,1:], reg_mod=condTS_mod, 
                                    y_eval=X[:,0], x_eval=X[:,1:], kern='gaussian', b=b)
        condTS_est[condTS_est < tau] = tau
        
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y.reshape(-1,1))
        NN_fit = train(mu, X_tensor, Y_tensor, lr=lr, n_epochs=n_iter)
        theta_hat = np.zeros((n, t_eval.shape[0]))
        mu_hat = np.zeros((n, t_eval.shape[0]))
        IPW_hat = np.zeros((n, t_eval.shape[0]))
        beta_hat = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            # Define the data matrix for evaluating the fitted regression model
            X_eval = np.column_stack([t_eval[i]*np.ones(n), X[:,1:]])
            X_eval_tensor = torch.from_numpy(X_eval)
            for j in range(X_eval.shape[0]):
                # Compute the gradient of the fitted regression model with respect to the first coordinate
                x = X_eval_tensor[j,:]
                x = x.clone().detach().requires_grad_(True)
                y = NN_fit(x)
                y_scalar = y[0]
                y_scalar.backward()
                x_grad = x.grad
                beta_hat[j,i] = x_grad[0].item()
            NN_fit.eval()
            mu_pred = NN_fit(X_eval_tensor)
            mu_hat[:,i] = mu_pred.detach().numpy()[:,0]
            
            if bnd_cor:
                if t_eval[i] < np.min(X[:,0]) or t_eval[i] > np.max(X[:,0]):
                    IPW_hat[:,i] = 0
                elif t_eval[i] < np.min(X[:,0]) + h:
                    alpha = (t_eval[i] - np.min(X[:,0]))/h
                    bndkern = BndKern((X[:,0] - t_eval[i])/h, kern=kern, deriv_ord=1, alpha=alpha, bnd='left')
                    IPW_hat[:,i] = ((X[:,0] - t_eval[i])/h) * bndkern * (Y - mu_hat[:,i] - (X[:,0] - t_eval[i])*beta_hat[:,i]) / (h**2 * condTS_est)
                elif t_eval[i] > np.max(X[:,0]) - h:
                    alpha = (np.max(X[:,0]) - t_eval[i])/h
                    bndkern = BndKern((X[:,0] - t_eval[i])/h, kern=kern, deriv_ord=1, alpha=alpha, bnd='right')
                    IPW_hat[:,i] = ((X[:,0] - t_eval[i])/h) * bndkern * (Y - mu_hat[:,i] - (X[:,0] - t_eval[i])*beta_hat[:,i]) / (h**2 * condTS_est)
                else:
                    IPW_hat[:,i] = ((X[:,0] - t_eval[i])/h) * kern((t_eval[i] - X[:,0])/h) * (Y - mu_hat[:,i] - (X[:,0] - t_eval[i])*beta_hat[:,i]) / (h**2 * sigmaK_sq * condTS_est)
            else:
                IPW_hat[:,i] = ((X[:,0] - t_eval[i])/h) * kern((t_eval[i] - X[:,0])/h) * (Y - mu_hat[:,i] - (X[:,0] - t_eval[i])*beta_hat[:,i]) / (h**2 * sigmaK_sq * condTS_est)
            
            # Add up the IPW and RA components
            theta_hat[:,i] = IPW_hat[:,i] + beta_hat[:,i]

        theta_est = np.mean(theta_hat, axis=0)
        
        # Estimate the variance of theta(t) using the square of the influence function
        var_est = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            var_est[:,i] = (IPW_hat[:,i] + (beta_hat[:,i] - theta_est[i]))**2 * (h**3)
        sd_est = np.sqrt(np.mean(var_est, axis=0)/(n*(h**3)))
    else:
        # Conduct L-fold cross-fittings: fit the reciprocal of the conditional model 
        # and the regression model on the training fold data and evaluate it on the test fold data
        kf = KFold(n_splits=L, shuffle=True, random_state=0)
        theta_hat = np.zeros((n, t_eval.shape[0]))
        mu_hat = np.zeros((n, t_eval.shape[0]))
        beta_hat = np.zeros((n, t_eval.shape[0]))
        IPW_hat = np.zeros((n, t_eval.shape[0]))
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
            
            X_tr_tensor = torch.from_numpy(X_tr)
            Y_tr_tensor = torch.from_numpy(Y_tr.reshape(-1,1))
            NN_fit = train(mu, X_tr_tensor, Y_tr_tensor, lr=lr, n_epochs=n_iter)
            for i in range(t_eval.shape[0]):
                # Define the data matrix for evaluating the fitted regression model
                X_eval_te = np.column_stack([t_eval[i]*np.ones(X_te.shape[0]), X_te[:,1:]])
                X_eval_te_tensor = torch.from_numpy(X_eval_te)
                for j in range(X_eval_te.shape[0]):
                    # Compute the gradient of the fitted regression model with respect to the first coordinate
                    x = X_eval_te_tensor[j,:]
                    x = x.clone().detach().requires_grad_(True)
                    y = NN_fit(x)
                    y_scalar = y[0]
                    y_scalar.backward()
                    x_grad = x.grad
                    beta_hat[j,i] = x_grad[0].item()
                    
                NN_fit.eval()
                mu_pred = NN_fit(X_eval_te_tensor)
                mu_hat[te_ind,i] = mu_pred.detach().numpy()[:,0]
                
                if bnd_cor:
                    if t_eval[i] < np.min(X[te_ind,0]) or t_eval[i] > np.max(X[te_ind,0]):
                        IPW_hat[te_ind,i] = 0
                    elif t_eval[i] < np.min(X[te_ind,0]) + h:
                        alpha = (t_eval[i] - np.min(X[te_ind,0]))/h
                        bndkern = BndKern((X[te_ind,0] - t_eval[i])/h, kern=kern, deriv_ord=1, alpha=alpha, bnd='left')
                        IPW_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * bndkern * (Y_te - mu_hat[te_ind,i] - (X[te_ind,0] - t_eval[i])*beta_hat[te_ind,i]) / (h**2 * condTS_est)
                    elif t_eval[i] > np.max(X[te_ind,0]) - h:
                        alpha = (np.max(X[te_ind,0]) - t_eval[i])/h
                        bndkern = BndKern((X[te_ind,0] - t_eval[i])/h, kern=kern, deriv_ord=1, alpha=alpha, bnd='right')
                        IPW_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * bndkern * (Y_te - mu_hat[te_ind,i] - (X[te_ind,0] - t_eval[i])*beta_hat[te_ind,i]) / (h**2 * condTS_est)
                    else:
                        IPW_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * kern((t_eval[i] - X[te_ind,0])/h) * (Y_te - mu_hat[te_ind,i] - (X[te_ind,0] - t_eval[i])*beta_hat[te_ind,i]) / (h**2 * sigmaK_sq * condTS_est)
                else:
                    IPW_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * kern((t_eval[i] - X[te_ind,0])/h) * (Y_te - mu_hat[te_ind,i] - (X[te_ind,0] - t_eval[i])*beta_hat[te_ind,i]) / (h**2 * sigmaK_sq * condTS_est) 
                    
                # Add up the IPW and RA components
                theta_hat[te_ind,i] = IPW_hat[te_ind,i] + beta_hat[te_ind,i]

        theta_est = np.mean(theta_hat, axis=0)
        
        # Estimate the variance of theta(t) using the square of the influence function
        var_est = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            var_est[:,i] = (IPW_hat[:,i] + (beta_hat[:,i] - theta_est[i]))**2 * (h**3)
        sd_est = np.sqrt(np.mean(var_est, axis=0)/(n*(h**3)))
    return theta_est, sd_est



def DRDRDerivSKLearn(Y, X, t_eval, mu, condTS_type, condTS_mod, L, h, kern, tau=0.01, 
                     b=None, bnd_cor=False, delta=0.01):
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
            condTS_est = CondDenEst(X[:,0], X[:,1:], reg_mod=condTS_mod, 
                                    y_eval=X[:,0], x_eval=X[:,1:], kern='gaussian', b=b)
        condTS_est[condTS_est < tau] = tau
        
        mu_fit = mu.fit(X, Y)
        
        theta_hat = np.zeros((n, t_eval.shape[0]))
        mu_hat = np.zeros((n, t_eval.shape[0]))
        IPW_hat = np.zeros((n, t_eval.shape[0]))
        t_new = np.linspace(np.min(t_eval)-delta, np.max(t_eval)+delta, t_eval.shape[0]+1)
        beta_hat = np.zeros((n, t_new.shape[0]))
        for i in range(t_new.shape[0]):
            X_new = np.column_stack([t_new[i]*np.ones(n), X[:,1:]])
            beta_hat[:,i] = mu_fit.predict(X_new)
        beta_hat = np.diff(beta_hat, axis=1)
        for i in range(t_eval.shape[0]):
            # Define the data matrix for evaluating the fitted regression model
            X_eval = np.column_stack([t_eval[i]*np.ones(n), X[:,1:]])
            mu_hat[:,i] = mu_fit.predict(X_eval)
            
            if bnd_cor:
                if t_eval[i] < np.min(X[:,0]) or t_eval[i] > np.max(X[:,0]):
                    IPW_hat[:,i] = 0
                elif t_eval[i] < np.min(X[:,0]) + h:
                    alpha = (t_eval[i] - np.min(X[:,0]))/h
                    bndkern = BndKern((X[:,0] - t_eval[i])/h, kern=kern, deriv_ord=1, alpha=alpha, bnd='left')
                    IPW_hat[:,i] = ((X[:,0] - t_eval[i])/h) * bndkern * (Y - mu_hat[:,i] - (X[:,0] - t_eval[i])*beta_hat[:,i]) / (h**2 * condTS_est)
                elif t_eval[i] > np.max(X[:,0]) - h:
                    alpha = (np.max(X[:,0]) - t_eval[i])/h
                    bndkern = BndKern((X[:,0] - t_eval[i])/h, kern=kern, deriv_ord=1, alpha=alpha, bnd='right')
                    IPW_hat[:,i] = ((X[:,0] - t_eval[i])/h) * bndkern * (Y - mu_hat[:,i] - (X[:,0] - t_eval[i])*beta_hat[:,i]) / (h**2 * condTS_est)
                else:
                    IPW_hat[:,i] = ((X[:,0] - t_eval[i])/h) * kern((t_eval[i] - X[:,0])/h) * (Y - mu_hat[:,i] - (X[:,0] - t_eval[i])*beta_hat[:,i]) / (h**2 * sigmaK_sq * condTS_est)
            else:
                IPW_hat[:,i] = ((X[:,0] - t_eval[i])/h) * kern((t_eval[i] - X[:,0])/h) * (Y - mu_hat[:,i] - (X[:,0] - t_eval[i])*beta_hat[:,i]) / (h**2 * sigmaK_sq * condTS_est)
            
            # Add up the IPW and RA components
            theta_hat[:,i] = IPW_hat[:,i] + beta_hat[:,i]

        theta_est = np.mean(theta_hat, axis=0)
        
        # Estimate the variance of theta(t) using the square of the influence function
        var_est = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            var_est[:,i] = (IPW_hat[:,i] + (beta_hat[:,i] - theta_est[i]))**2 * (h**3)
        sd_est = np.sqrt(np.mean(var_est, axis=0)/(n*(h**3)))
    else:
        # Conduct L-fold cross-fittings: fit the reciprocal of the conditional model 
        # and the regression model on the training fold data and evaluate it on the test fold data
        kf = KFold(n_splits=L, shuffle=True, random_state=0)
        theta_hat = np.zeros((n, t_eval.shape[0]))
        mu_hat = np.zeros((n, t_eval.shape[0]))
        beta_hat = np.zeros((n, t_eval.shape[0]))
        
        t_new = np.linspace(np.min(t_eval)-delta, np.max(t_eval)+delta, t_eval.shape[0]+1)
        beta_can = np.zeros((n, t_new.shape[0]))
        IPW_hat = np.zeros((n, t_eval.shape[0]))
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
            for i in range(t_new.shape[0]):
                X_new_te = np.column_stack([t_new[i]*np.ones(X_te.shape[0]), X_te[:,1:]])
                beta_can[te_ind,i] = mu_fit.predict(X_new_te)
            beta_hat[te_ind,:] = np.diff(beta_can[te_ind,:], axis=1)
            
            for i in range(t_eval.shape[0]):
                # Define the data matrix for evaluating the fitted regression model
                X_eval_te = np.column_stack([t_eval[i]*np.ones(X_te.shape[0]), X_te[:,1:]])
                mu_hat[te_ind,i] = mu_fit.predict(X_eval_te)
                
                if bnd_cor:
                    if t_eval[i] < np.min(X[te_ind,0]) or t_eval[i] > np.max(X[te_ind,0]):
                        IPW_hat[te_ind,i] = 0
                    elif t_eval[i] < np.min(X[te_ind,0]) + h:
                        alpha = (t_eval[i] - np.min(X[te_ind,0]))/h
                        bndkern = BndKern((X[te_ind,0] - t_eval[i])/h, kern=kern, deriv_ord=1, alpha=alpha, bnd='left')
                        IPW_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * bndkern * (Y_te - mu_hat[te_ind,i] - (X[te_ind,0] - t_eval[i])*beta_hat[te_ind,i]) / (h**2 * condTS_est)
                    elif t_eval[i] > np.max(X[te_ind,0]) - h:
                        alpha = (np.max(X[te_ind,0]) - t_eval[i])/h
                        bndkern = BndKern((X[te_ind,0] - t_eval[i])/h, kern=kern, deriv_ord=1, alpha=alpha, bnd='right')
                        IPW_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * bndkern * (Y_te - mu_hat[te_ind,i] - (X[te_ind,0] - t_eval[i])*beta_hat[te_ind,i]) / (h**2 * condTS_est)
                    else:
                        IPW_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * kern((t_eval[i] - X[te_ind,0])/h) * (Y_te - mu_hat[te_ind,i] - (X[te_ind,0] - t_eval[i])*beta_hat[te_ind,i]) / (h**2 * sigmaK_sq * condTS_est)
                else:
                    IPW_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * kern((t_eval[i] - X[te_ind,0])/h) * (Y_te - mu_hat[te_ind,i] - (X[te_ind,0] - t_eval[i])*beta_hat[te_ind,i]) / (h**2 * sigmaK_sq * condTS_est) 
                    
                # Add up the IPW and RA components
                theta_hat[te_ind,i] = IPW_hat[te_ind,i] + beta_hat[te_ind,i]

        theta_est = np.mean(theta_hat, axis=0)
        
        # Estimate the variance of theta(t) using the square of the influence function
        var_est = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            var_est[:,i] = (IPW_hat[:,i] + (beta_hat[:,i] - theta_est[i]))**2 * (h**3)
        sd_est = np.sqrt(np.mean(var_est, axis=0)/(n*(h**3)))
    return theta_est, sd_est


def DRDerivCurve(Y, X, t_eval=None, est="RA", beta_mod=None, n_iter=1000, lr=0.1, 
                 condTS_type=None, condTS_mod=None, tau=0.01, L=1, h=None, kern="epanechnikov", 
                 h_cond=None, print_bw=True, delta=0.01, self_norm=True, bnd_cor=False):
    '''
    Dose-response curve derivative estimation under the positivity condition.
    
    Parameters
    ----------
        Y: (n,)-array
            The outcomes of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points. (Default: t_eval=None. 
            Then, t_eval=X[:,0], which consists of the observed treatment variables.)
            
        est: str
            The type of the dose-response curve estimator. (Default: est="RA". 
            Other choices include "IPW" and "DR".)
            
        h: float
            The bandwidth parameter for the IPW/DR estimator.
            
        print_bw: boolean
            The indicator of whether the current bandwidth parameters should be
            printed to the console. (Default: print_bw=True.)
            
        kern: str
            The name of the kernel function for the IPW/DR estimator.
            (Default: "epanechnikov".)
    
    Return
    ----------
        theta_est: (m,)-array
            The estimated derivatives of the dose-response curve evaluated 
            at points "t_eval".
            
        sd_est: (m,)-array (if est="DR")
            The estimated standard error of the DR derivative estimator 
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
        if isinstance(beta_mod, BaseEstimator):
            theta_est = RADRDerivSKLearn(Y, X, t_eval, beta_mod, L, delta)
        else:
            theta_est = RADRDeriv(Y, X, t_eval, beta_mod, L, n_iter, lr)
    elif est == "IPW":
        theta_est = IPWDRDeriv(Y, X, t_eval, condTS_type, condTS_mod, L, h, kern, tau, h_cond, 
                               self_norm, bnd_cor)
    elif isinstance(beta_mod, BaseEstimator):
        theta_est = DRDRDerivSKLearn(Y, X, t_eval, beta_mod, condTS_type, condTS_mod, L, h, kern, 
                                     tau, h_cond, bnd_cor, delta)
    else:
        theta_est = DRDRDeriv(Y, X, t_eval, beta_mod, condTS_type, condTS_mod, L, h, kern, n_iter, lr, 
                              tau, h_cond, bnd_cor)
    
    return theta_est
