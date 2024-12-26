# -*- coding: utf-8 -*-

# Author: Yikun Zhang
# Last Editing: Dec 22, 2024

# Description: This script contains the implementations of the IPW and doubly 
# robust estimators of the derivative of a dose-response curve with and without 
# the positivity condition.

import numpy as np
from rbf import KernelRetrieval
from utils1 import CondDenEst, CondDenEstKDE, KDE
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator

import torch
import torch.nn as nn
import torch.optim as optim

#=======================================================================================#
# Implementations of the proposed estimators that assume the positivity condition

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
    

def train(mod, X_train, Y_train, lr=0.01, n_epochs=10, momentum=0.7, weight_decay=0,
          print_loss=True):
    '''
    Utility function for training the PyTorch neural network model via stochastic
    gradient descent.
    
    Parameters
    ----------
        mod: python class
            The neural network class defined by PyTorch.
            
        X_train: (n,d+1)-torch.Tensor
            The first column of "X_train" is the treatment/exposure variable, 
            while the other d columns are the confounding variables of n observations.
            
        Y_train: (n,)-torch.Tensor
            The outcome variables of n observations.
            
        lr: float
            The learning rate (Default: lr=0.01.)
            
        n_epochs: int
            The number of training epochs. (Default: n_epochs=10.)
            
        momentum: float
            The momentum factor (Default: momentum=0.7.)
            
        weight_decay: float
            The weight decay (L2 penalty) (Default: weight_decay=0.)
            
        print_loss: boolean
            An indicator of whether the training loss will be printed to the console.
            
    Return
    ----------
        model: python object
            The fitted model instance of a neural network class defined by PyTorch.
    '''
    # Initialize the model, loss function, and optimizer
    model = mod(input_size=X_train.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, 
                          weight_decay=weight_decay)
    
    for epoch in range(n_epochs):
        model.train()
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()    # Zero the gradients
        loss.backward()          # Backpropagate
        optimizer.step()         # Update weights
        if print_loss:
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')
        
    return model


def RADRDeriv(Y, X, t_eval, mu, L=1, n_iter=1000, lr=0.1, multi_boot=False, B=1000):
    '''
    Estimating the derivative of a dose-response curve through the regression 
    adjustment (or G-computation) form by a PyTorch neural network model under 
    the positivity condition.
    
    Parameters
    ----------
        Y: (n,)-array
            The outcome variables of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are the confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points.
            
        mu: a neural network class defined by PyTorch
            The conditional mean outcome (or regression) model of Y given X.
            
        L: int
            The number of data folds for cross-fitting. When L<= 1, no cross-fittings 
            are applied and the regression model is fitted on the entire dataset.
            (Default: L=1.)
            
        n_iter: int
            The number of iterations or training epochs of the neural network model.
            (Default: n_iter=1000.)
            
        lr: float
            The learning rate (Default: lr=0.01.)
            
        multi_boot: boolean
            An indicator of whether the multiplier bootstrap will be run. 
            (Default: multi_boot=False.)
            
        B: int
            The number of bootstrapping times. (Default: B=1000.)
            
    Return
    ----------
        theta_est: (m,)-array
            The estimated derivative of the dose-response curve evaluated at 
            points "t_eval".
            
        mu_boot: (B,m)-array
            The estimated derivatives of the dose-response curves on bootstrapping 
            data evaluated at points "t_eval". (Only return this quantity when 
            "multi_boot=True".)
    '''
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
    '''
    Estimating the derivative of a dose-response curve through the regression 
    adjustment (or G-computation) form under the positivity condition.
    
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
            
        delta: float
            The step value for computing the finite differences (or numerical partial
            differentiation) of the fitted regression model.
            
    Return
    ----------
        theta_est: (m,)-array
            The estimated derivative of the dose-response curve evaluated at 
            points "t_eval".
    '''
    
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


def IPWDRDeriv(Y, X, t_eval, condTS_type, condTS_mod, L, h, kern="epanechnikov", 
               tau=0.001, b=None, self_norm=True):
    '''
    Estimating the derivative of a dose-response curve through the inverse 
    probability weighting (IPW) form under the positivity condition.
    
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
            
        condTS_mod: cikit-learn model or any python model that can use ".fit()" and ".predict()"
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
        theta_est: (m,)-array
            The estimated derivative of the dose-response curve evaluated at 
            points "t_eval".
    '''
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


def DRDRDeriv(Y, X, t_eval, mu, condTS_type, condTS_mod, L, h, kern="epanechnikov", 
              n_iter=1000, lr=0.01, tau=0.001, b=None, self_norm=True):
    '''
    Estimating the derivative of a dose-response curve through the doubly robust
    (DR) form by a PyTorch neural network model under the positivity condition.
    
    Parameters
    ----------
        Y: (n,)-array
            The outcome variables of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are the confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points.
            
        mu: a neural network class defined by PyTorch
            The conditional mean outcome (or regression) model of Y given X.
            
        condTS_type: str
            Specifying the model type for estimating the conditional density of
            the treatment variable T given the covariate vector S.
            
        condTS_mod: cikit-learn model or any python model that can use ".fit()" and ".predict()"
            The regression model for estimating the conditional density of T given S.
            
        L: int
            The number of data folds for cross-fitting. When L<= 1, no cross-fittings 
            are applied and the regression model is fitted on the entire dataset.
            (Default: L=1.)
            
        h: float
            The bandwidth parameter.
            
        kern: str
            The name of the kernel function. (Default: kern="epanechnikov".)
            
        n_iter: int
            The number of iterations or training epochs of the neural network model.
            (Default: n_iter=1000.)
            
        lr: float
            The learning rate (Default: lr=0.01.)
            
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
        theta_est: (m,)-array
            The estimated derivative of the dose-response curve evaluated at 
            points "t_eval".
            
        sd_est: (m,)-array
            The estimated asymptotic stdndard deviation of the DR derivative 
            estimator evaluated at points "t_eval".
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
            condTS_est = CondDenEst(X[:,0], X[:,1:], reg_mod=condTS_mod, 
                                    y_eval=X[:,0], x_eval=X[:,1:], kern='gaussian', b=b)
        condTS_est[condTS_est < tau] = tau
        
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y.reshape(-1,1))
        NN_fit = train(mu, X_tensor, Y_tensor, lr=lr, n_epochs=n_iter)
        mu_hat = np.zeros((n, t_eval.shape[0]))
        IPW_hat = np.zeros((n, t_eval.shape[0]))
        beta_hat = np.zeros((n, t_eval.shape[0]))
        norm_w = np.zeros((t_eval.shape[0],))
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
            
            IPW_hat[:,i] = ((X[:,0] - t_eval[i])/h) * kern((t_eval[i] - X[:,0])/h) * (Y - mu_hat[:,i] - (X[:,0] - t_eval[i])*beta_hat[:,i]) / (h**2 * sigmaK_sq * condTS_est)
            # Self-normalizing weights
            norm_w[i] = np.sum(kern((t_eval[i] - X[:,0])/h) / condTS_est) / (n * h)
        
        if self_norm:
            IPW_hat = IPW_hat / norm_w
        # Add up the IPW and RA components
        theta_hat = IPW_hat + beta_hat
        theta_est = np.mean(theta_hat, axis=0, where=~np.isnan(theta_hat))
        
        # Estimate the variance of theta(t) using the square of the influence function
        var_est = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            var_est[:,i] = (IPW_hat[:,i] + (beta_hat[:,i] - theta_est[i]))**2 * (h**3)
        sd_est = np.sqrt(np.mean(var_est, axis=0)/(n*(h**3)))
    else:
        # Conduct L-fold cross-fittings: fit the reciprocal of the conditional model 
        # and the regression model on the training fold data and evaluate it on the test fold data
        kf = KFold(n_splits=L, shuffle=True, random_state=0)
        mu_hat = np.zeros((n, t_eval.shape[0]))
        beta_hat = np.zeros((n, t_eval.shape[0]))
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
                
                IPW_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * kern((t_eval[i] - X[te_ind,0])/h) * (Y_te - mu_hat[te_ind,i] - (X[te_ind,0] - t_eval[i])*beta_hat[te_ind,i]) / (h**2 * sigmaK_sq * condTS_est) 
                
                # Self-normalizing weights
                w = np.sum(kern((t_eval[i] - X[te_ind,0])/h) / condTS_est) / (n * h)
                norm_w[i] = norm_w[i] + w
        
        if self_norm:
            IPW_hat = IPW_hat / norm_w
        # Add up the IPW and RA components
        theta_hat = IPW_hat + beta_hat
        theta_est = np.mean(theta_hat, axis=0, where=~np.isnan(theta_hat))
        
        # Estimate the variance of theta(t) using the square of the influence function
        var_est = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            var_est[:,i] = (IPW_hat[:,i] + (beta_hat[:,i] - theta_est[i]))**2 * (h**3)
        sd_est = np.sqrt(np.mean(var_est, axis=0)/(n*(h**3)))
    return theta_est, sd_est



def DRDRDerivSKLearn(Y, X, t_eval, mu, condTS_type, condTS_mod, L, h, kern="epanechnikov", 
                     tau=0.001, b=None, delta=0.01, self_norm=True):
    '''
    Estimating the derivative of a dose-response curve through the doubly robust
    (DR) form under the positivity condition.
    
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
            
        condTS_mod: cikit-learn model or any python model that can use ".fit()" and ".predict()"
            The regression model for estimating the conditional density of T given S.
            
        L: int
            The number of data folds for cross-fitting. When L<= 1, no cross-fittings 
            are applied and the regression model is fitted on the entire dataset.
            (Default: L=1.)
            
        h: float
            The bandwidth parameter.
            
        kern: str
            The name of the kernel function. (Default: kern="epanechnikov".)
            
        n_iter: int
            The number of iterations or training epochs of the neural network model.
            (Default: n_iter=1000.)
            
        lr: float
            The learning rate (Default: lr=0.01.)
            
        tau: float
            The threshold value that lower bounds the estimated conditional density
            values. (Default: tau=0.001.)
            
        b: float
            The bandwidth parameter for the kernel-smoothed conditional density
            estimation methods. (Default: b=None.)
        
        delta: float
            The step value for computing the finite differences (or numerical partial
            differentiation) of the fitted regression model.
        
        self_norm: boolean
            An indicator of whether the self-normalized version is implemented.
            (Default: self_norm=True.)
            
    Return
    ----------
        theta_est: (m,)-array
            The estimated derivative of the dose-response curve evaluated at 
            points "t_eval".
            
        sd_est: (m,)-array
            The estimated asymptotic stdndard deviation of the DR derivative 
            estimator evaluated at points "t_eval".
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
            condTS_est = CondDenEst(X[:,0], X[:,1:], reg_mod=condTS_mod, 
                                    y_eval=X[:,0], x_eval=X[:,1:], kern='gaussian', b=b)
        condTS_est[condTS_est < tau] = tau
        
        mu_fit = mu.fit(X, Y)
        
        mu_hat = np.zeros((n, t_eval.shape[0]))
        IPW_hat = np.zeros((n, t_eval.shape[0]))
        t_new = np.linspace(np.min(t_eval)-delta, np.max(t_eval)+delta, t_eval.shape[0]+1)
        beta_hat = np.zeros((n, t_new.shape[0]))
        norm_w = np.zeros((t_eval.shape[0],))
        for i in range(t_new.shape[0]):
            X_new = np.column_stack([t_new[i]*np.ones(n), X[:,1:]])
            beta_hat[:,i] = mu_fit.predict(X_new)
        beta_hat = np.diff(beta_hat, axis=1)
        for i in range(t_eval.shape[0]):
            # Define the data matrix for evaluating the fitted regression model
            X_eval = np.column_stack([t_eval[i]*np.ones(n), X[:,1:]])
            mu_hat[:,i] = mu_fit.predict(X_eval)
            
            IPW_hat[:,i] = ((X[:,0] - t_eval[i])/h) * kern((t_eval[i] - X[:,0])/h) * (Y - mu_hat[:,i] - (X[:,0] - t_eval[i])*beta_hat[:,i]) / (h**2 * sigmaK_sq * condTS_est)
            # Self-normalizing weights
            norm_w[i] = np.sum(kern((t_eval[i] - X[:,0])/h) / condTS_est) / (n * h)

        if self_norm:
            IPW_hat = IPW_hat / norm_w
        # Add up the IPW and RA components
        theta_hat = IPW_hat + beta_hat
        theta_est = np.mean(theta_hat, axis=0, where=~np.isnan(theta_hat))
        
        # Estimate the variance of theta(t) using the square of the influence function
        var_est = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            var_est[:,i] = (IPW_hat[:,i] + (beta_hat[:,i] - theta_est[i]))**2 * (h**3)
        sd_est = np.sqrt(np.mean(var_est, axis=0)/(n*(h**3)))
    else:
        # Conduct L-fold cross-fittings: fit the reciprocal of the conditional model 
        # and the regression model on the training fold data and evaluate it on the test fold data
        kf = KFold(n_splits=L, shuffle=True, random_state=0)
        mu_hat = np.zeros((n, t_eval.shape[0]))
        beta_hat = np.zeros((n, t_eval.shape[0]))
        
        t_new = np.linspace(np.min(t_eval)-delta, np.max(t_eval)+delta, t_eval.shape[0]+1)
        beta_can = np.zeros((n, t_new.shape[0]))
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
            for i in range(t_new.shape[0]):
                X_new_te = np.column_stack([t_new[i]*np.ones(X_te.shape[0]), X_te[:,1:]])
                beta_can[te_ind,i] = mu_fit.predict(X_new_te)
            beta_hat[te_ind,:] = np.diff(beta_can[te_ind,:], axis=1)
            
            for i in range(t_eval.shape[0]):
                # Define the data matrix for evaluating the fitted regression model
                X_eval_te = np.column_stack([t_eval[i]*np.ones(X_te.shape[0]), X_te[:,1:]])
                mu_hat[te_ind,i] = mu_fit.predict(X_eval_te)
                
                IPW_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * kern((t_eval[i] - X[te_ind,0])/h) * (Y_te - mu_hat[te_ind,i] - (X[te_ind,0] - t_eval[i])*beta_hat[te_ind,i]) / (h**2 * sigmaK_sq * condTS_est) 
                    
                # Self-normalizing weights
                w = np.sum(kern((t_eval[i] - X[te_ind,0])/h) / condTS_est) / (n * h)
                norm_w[i] = norm_w[i] + w

        if self_norm:
            IPW_hat = IPW_hat / norm_w
        # Add up the IPW and RA components
        theta_hat = IPW_hat + beta_hat
        theta_est = np.mean(theta_hat, axis=0, where=~np.isnan(theta_hat))
        
        # Estimate the variance of theta(t) using the square of the influence function
        var_est = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            var_est[:,i] = (IPW_hat[:,i] + (beta_hat[:,i] - theta_est[i]))**2 * (h**3)
        sd_est = np.sqrt(np.mean(var_est, axis=0)/(n*(h**3)))
    return theta_est, sd_est


def DRDerivCurve(Y, X, t_eval=None, est="RA", beta_mod=None, n_iter=1000, lr=0.01, 
                 condTS_type=None, condTS_mod=None, L=1, h=None, kern="epanechnikov", 
                 tau=0.001, h_cond=None, delta=0.01, self_norm=True, print_bw=True):
    '''
    Dose-response curve derivative estimation under the positivity condition.
    
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
            
        beta_mod: PyTorch neural network class or scikit-learn model or any python 
        model that can use ".fit()" and ".predict()"
            The conditional mean outcome (or regression) model of Y given X.
            
        n_iter: int
            The number of iterations or training epochs of the neural network model.
            (Default: n_iter=1000.)
            
        lr: float
            The learning rate (Default: lr=0.01.)
            
        condTS_type: str
            Specifying the model type for estimating the conditional density of
            the treatment variable T given the covariate vector S.
            
        condTS_mod: cikit-learn model or any python model that can use ".fit()" and ".predict()"
            The regression model for estimating the conditional density of T given S.
            
        L: int
            The number of data folds for cross-fitting. When L<= 1, no cross-fittings 
            are applied and the regression model is fitted on the entire dataset.
            (Default: L=1.)
            
        h: float
            The bandwidth parameter for the IPW/DR estimator. (Default: h=None. 
            Then the Silverman's rule of thumb is applied; see Chen et al.(2016) 
            for details.)
            
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
        theta_est: (m,)-array
            The estimated derivative of the dose-response curve evaluated at 
            points "t_eval".
            
        sd_est: (m,)-array (if est="DR")
            The estimated asymptotic stdndard deviation of the DR derivative 
            estimator evaluated at points "t_eval".
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
        theta_est = IPWDRDeriv(Y, X, t_eval, condTS_type, condTS_mod, L, h, kern, 
                               tau, h_cond, self_norm)
    elif isinstance(beta_mod, BaseEstimator):
        theta_est = DRDRDerivSKLearn(Y, X, t_eval, beta_mod, condTS_type, condTS_mod, 
                                     L, h, kern, tau, h_cond, delta, self_norm)
    else:
        theta_est = DRDRDeriv(Y, X, t_eval, beta_mod, condTS_type, condTS_mod, L, 
                              h, kern, n_iter, lr, tau, h_cond, self_norm)
    
    return theta_est




#=======================================================================================#
# Implementations of the proposed estimators without assuming the positivity 
# condition (work for additive confounding models)

def RADRDerivBC(Y, X, t_eval, mu, L=1, n_iter=1000, lr=0.01, h_bar=None, 
                kernT_bar="gaussian", print_bw=False):
    '''
    Estimating the derivative of a dose-response curve through the regression 
    adjustment (or G-computation) form by a PyTorch neural network model without 
    assuming the positivity condition.
    
    Parameters
    ----------
        Y: (n,)-array
            The outcome variables of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are the confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points.
            
        mu: a neural network class defined by PyTorch
            The conditional mean outcome (or regression) model of Y given X.
            
        L: int
            The number of data folds for cross-fitting. When L<= 1, no cross-fittings 
            are applied and the regression model is fitted on the entire dataset.
            (Default: L=1.)
            
        n_iter: int
            The number of iterations or training epochs of the neural network model.
            (Default: n_iter=1000.)
            
        lr: float
            The learning rate (Default: lr=0.01.)
            
        h_bar: float
            The bandwidth parameters for the Nadaraya-Watson conditional CDF estimator. 
            (Default: h_bar=None. Then, the Silverman's rule of thumb is applied. 
            See Chen et al.(2016) for details.)
            
        kernT_bar: str
            The name of the kernel function for Nadaraya-Watson conditional CDF 
            estimator. (Default: "gaussian".)
            
        print_bw: boolean
            The indicator of whether the current bandwidth parameters should be
            printed to the console. (Default: print_bw=False.)
            
    Return
    ----------
        theta_C: (m,)-array
            The estimated derivative of the dose-response curve evaluated at 
            points "t_eval".
    '''
    n = X.shape[0]  ## Number of data points
    if h_bar is None:
        # Apply the Silverman's rule of thumb bandwidth in Chen et al. (2016)
        h_bar = (4/3)**(1/5)*(n**(-1/5))*np.std(X[:,0])
    if print_bw:
        print("The current bandwidth for the conditional CDF estimator is "+ str(h_bar) + ".\n")
    
    # Compute the weight matrix for NW conditional CDF estimator
    kernT_bar, sigmaK_sq, K_sq = KernelRetrieval(kernT_bar)
    weight_mat = kernT_bar((t_eval - X[:,0].reshape(-1,1)) / h_bar)
    weight_mat = weight_mat / np.sum(weight_mat, axis=0)
    weight_mat[np.isnan(weight_mat)] = 0
    
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
            
        theta_C = np.sum(beta_est * weight_mat, axis=0)
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
                    
        theta_C = np.sum(beta_est * weight_mat, axis=0)
    return theta_C


def IPWDRDerivBC(Y, X, t_eval, L=1, h=None, kern="epanechnikov", b=None, 
                 thres_val=0.75, self_norm=True):
    '''
    Estimating the derivative of a dose-response curve through the inverse 
    probability weighting (IPW) form without assuming the positivity condition.
    
    Parameters
    ----------
        Y: (n,)-array
            The outcome variables of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are the confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points.
            
        L: int
            The number of data folds for cross-fitting. When L<= 1, no cross-fittings 
            are applied and the regression model is fitted on the entire dataset.
            (Default: L=1.)
            
        h: float
            The bandwidth parameter. (Default: h=None. Then, the Silverman's 
            rule of thumb is applied; see Chen et al.(2016) for details.)
            
        kern: str
            The name of the kernel function. (Default: kern="epanechnikov".)
            
        b: float
            The bandwidth parameter for the kernel-smoothed conditional density
            estimation methods. (Default: b=None.)
            
        thres_val: float
            The threshold factor that is multiplied to the maximum conditional 
            density values of S given T evaluated at the sample points. (Default: 
            thres_val=0.75.)
        
        self_norm: boolean
            An indicator of whether the self-normalized version is implemented.
            (Default: self_norm=True.)
            
    Return
    ----------
        theta_est: (m,)-array
            The estimated derivative of the dose-response curve evaluated at 
            points "t_eval".
    '''
    kern_type = kern
    kern, sigmaK_sq, K_sq = KernelRetrieval(kern)
    n = X.shape[0]  ## Number of data points
    if h is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter
        h = (4/3)**(1/5)*(n**(-1/5))*np.std(X[:,0])
        print("The current bandwidth is "+ str(h) + ".\n")
        
    if L <= 1:
        # No cross-fittings: fit the density models on the entire data
        kde_joint = KDE(X, data=X, kern='gaussian', h=b)
        
        beta_hat = np.zeros((n, t_eval.shape[0]))
        norm_w = np.zeros((t_eval.shape[0],))
        condST_full = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            # Define the data matrix for evaluating the interior densities
            X_eval = np.column_stack([t_eval[i]*np.ones(n), X[:,1:]])
            new_joint = KDE(X_eval, data=X, kern=kern_type, h=b)
            t_den = KDE(t_eval[i], data=X[:,0], kern=kern_type, h=b)[0]
            condST_den = new_joint / t_den
            condST_full[:,i] = condST_den
            # condST_den = condST_den * (condST_den >= np.quantile(condST_den, thres_val)) / (1 - thres_val)
            trunc_perc = np.mean(condST_den >= thres_val * np.max(condST_den))
            condST_den = condST_den * (condST_den >= thres_val * np.max(condST_den)) / (1 - trunc_perc)
            den_fac = condST_den / kde_joint
            
            # Self-normalizing weights
            norm_w[i] = np.sum(kern((t_eval[i] - X[:,0])/h) * den_fac) / h
            beta_hat[:,i] = ((X[:,0] - t_eval[i])/h) * kern((t_eval[i] - X[:,0])/h) * Y * den_fac/ (h**2 * sigmaK_sq)

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
        condST_full = np.zeros((n, t_eval.shape[0]))
        for tr_ind, te_ind in kf.split(X):
            X_tr = X[tr_ind,:]
            X_te = X[te_ind,:]
            Y_te = Y[te_ind]
            
            kde_joint = KDE(X_te, data=X_tr, kern='gaussian', h=b)
            for i in range(t_eval.shape[0]):
                # Define the data matrix for evaluating the interior densities
                X_eval_te = np.column_stack([t_eval[i]*np.ones(X_te.shape[0]), X_te[:,1:]])
                new_joint = KDE(X_eval_te, data=X_tr, kern='gaussian', h=b)
                t_den = KDE(t_eval[i], data=X_tr[:,0], kern='gaussian', h=b)[0]
                condST_den = new_joint / t_den
                condST_full[te_ind,i] = condST_den
                # condST_den = condST_den * (condST_den >= np.quantile(condST_den, thres_val)) / (1 - thres_val)
                trunc_perc = np.mean(condST_den >= thres_val * np.max(condST_den))
                condST_den = condST_den * (condST_den >= thres_val * np.max(condST_den)) / (1 - trunc_perc)
                den_fac = condST_den / kde_joint
                # Self-normalizing weights
                w = np.sum(kern((t_eval[i] - X_te[:,0])/h) * den_fac) / h
                if ~np.isnan(w) and w != np.inf:
                    norm_w[i] = norm_w[i] + w
                beta_hat[te_ind,i] = ((X_te[:,0] - t_eval[i])/h) * kern((t_eval[i] - X_te[:,0])/h) * Y_te * den_fac / (h**2 * sigmaK_sq)

        if self_norm:
            norm_w[norm_w == 0] = 1
            theta_est = np.sum(beta_hat, axis=0) / norm_w
        else:
            theta_est = np.mean(beta_hat, axis=0)
    return theta_est, condST_full


def DRDRDerivBC(Y, X, t_eval, mu, L=1, h=None, kern="epanechnikov", n_iter=1000, 
                lr=0.01, b=None, thres_val=0.75, self_norm=True):
    '''
    Estimating the derivative of a dose-response curve through the doubly robust 
    (DR) form by a PyTorch neural network model without assuming the positivity condition.
    
    Parameters
    ----------
        Y: (n,)-array
            The outcome variables of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are the confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points.
        
        mu: a neural network class defined by PyTorch
            The conditional mean outcome (or regression) model of Y given X.
            
        L: int
            The number of data folds for cross-fitting. When L<= 1, no cross-fittings 
            are applied and the regression model is fitted on the entire dataset.
            (Default: L=1.)
            
        h: float
            The bandwidth parameter. (Default: h=None. Then, the Silverman's 
            rule of thumb is applied; see Chen et al.(2016) for details.)
            
        kern: str
            The name of the kernel function. (Default: kern="epanechnikov".)
            
        n_iter: int
            The number of iterations or training epochs of the neural network model.
            (Default: n_iter=1000.)
            
        lr: float
            The learning rate (Default: lr=0.01.)
            
        b: float
            The bandwidth parameter for the kernel-smoothed conditional density
            estimation methods. (Default: b=None.)
            
        thres_val: float
            The threshold factor that is multiplied to the maximum conditional 
            density values of S given T evaluated at the sample points. (Default: 
            thres_val=0.75.)
        
        self_norm: boolean
            An indicator of whether the self-normalized version is implemented.
            (Default: self_norm=True.)
            
    Return
    ----------
        theta_est: (m,)-array
            The estimated derivative of the dose-response curve evaluated at 
            points "t_eval".
            
        sd_est: (m,)-array
            The estimated asymptotic stdndard deviation of the DR derivative 
            estimator evaluated at points "t_eval".
    '''
    kern_type = kern
    kern, sigmaK_sq, K_sq = KernelRetrieval(kern)
    n = X.shape[0]  ## Number of data points
    if h is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter
        h = (4/3)**(1/5)*(n**(-1/5))*np.std(X[:,0])
        print("The current bandwidth is "+ str(h) + ".\n")
        
    if L <= 1:
        # No cross-fittings: fit the density and regression model on the entire data
        kde_joint = KDE(X, data=X, kern='gaussian', h=b)
        
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y.reshape(-1,1))
        NN_fit = train(mu, X_tensor, Y_tensor, lr=lr, n_epochs=n_iter)
        mu_hat = np.zeros((n, t_eval.shape[0]))
        IPW_hat = np.zeros((n, t_eval.shape[0]))
        beta_hat = np.zeros((n, t_eval.shape[0]))
        condST_int = np.zeros((n, t_eval.shape[0]))
        norm_w = np.zeros((t_eval.shape[0],))
        for i in range(t_eval.shape[0]):
            # Define the data matrix for evaluating the interior densities and fitted regression model
            X_eval = np.column_stack([t_eval[i]*np.ones(n), X[:,1:]])
            new_joint = KDE(X_eval, data=X, kern=kern_type, h=b)
            t_den = KDE(t_eval[i], data=X[:,0], kern=kern_type, h=b)[0]
            condST_den = new_joint / t_den
            # condST_den = condST_den * (condST_den >= np.quantile(condST_den, thres_val)) / (1 - thres_val)
            trunc_perc = np.mean(condST_den >= thres_val * np.max(condST_den))
            condST_den = condST_den * (condST_den >= thres_val * np.max(condST_den)) / (1 - trunc_perc)
            den_fac = condST_den / kde_joint
            
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
            
            IPW_hat[:,i] = ((X[:,0] - t_eval[i])/h) * kern((t_eval[i] - X[:,0])/h) * (Y - mu_hat[:,i] - (X[:,0] - t_eval[i])*beta_hat[:,i]) * den_fac/ (h**2 * sigmaK_sq)
            
            # Self-normalizing weights
            norm_w[i] = np.sum(kern((t_eval[i] - X[:,0])/h) * den_fac) / (n * h)
            condST_int[:,i] = condST_den
            
        if self_norm:
            IPW_hat = IPW_hat / norm_w
        # Add up the IPW and RA components
        theta_hat = IPW_hat + beta_hat * condST_int
        theta_est = np.mean(theta_hat, axis=0, where=~np.isnan(theta_hat))
        
        # Estimate the variance of theta(t) using the square of the influence function
        var_est = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            var_est[:,i] = (IPW_hat[:,i] + (np.mean(beta_hat[:,i] * condST_int[:,i]) - theta_est[i]))**2 * (h**3)
        sd_est = np.sqrt(np.mean(var_est, axis=0)/(n*(h**3)))
    else:
        # Conduct L-fold cross-fittings: fit the reciprocal of the conditional model 
        # and the regression model on the training fold data and evaluate it on the test fold data
        kf = KFold(n_splits=L, shuffle=True, random_state=0)
        mu_hat = np.zeros((n, t_eval.shape[0]))
        beta_hat = np.zeros((n, t_eval.shape[0]))
        IPW_hat = np.zeros((n, t_eval.shape[0]))
        condST_int = np.zeros((n, t_eval.shape[0]))
        norm_w = np.zeros((t_eval.shape[0],))
        for tr_ind, te_ind in kf.split(X):
            X_tr = X[tr_ind,:]
            Y_tr = Y[tr_ind]
            X_te = X[te_ind,:]
            Y_te = Y[te_ind]
            
            kde_joint = KDE(X_te, data=X_tr, kern='gaussian', h=b)
            
            X_tr_tensor = torch.from_numpy(X_tr)
            Y_tr_tensor = torch.from_numpy(Y_tr.reshape(-1,1))
            NN_fit = train(mu, X_tr_tensor, Y_tr_tensor, lr=lr, n_epochs=n_iter)
            for i in range(t_eval.shape[0]):
                # Define the data matrix for evaluating the interior densities and fitted regression model
                X_eval_te = np.column_stack([t_eval[i]*np.ones(X_te.shape[0]), X_te[:,1:]])
                new_joint = KDE(X_eval_te, data=X_tr, kern='gaussian', h=b)
                t_den = KDE(t_eval[i], data=X_tr[:,0], kern='gaussian', h=b)[0]
                condST_den = new_joint / t_den
                # condST_den = condST_den * (condST_den >= np.quantile(condST_den, thres_val)) / (1 - thres_val)
                trunc_perc = np.mean(condST_den >= thres_val * np.max(condST_den))
                condST_den = condST_den * (condST_den >= thres_val * np.max(condST_den)) / (1 - trunc_perc)
                den_fac = condST_den / kde_joint
                
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
                
                IPW_hat[te_ind,i] = ((X[te_ind,0] - t_eval[i])/h) * kern((t_eval[i] - X[te_ind,0])/h) * (Y_te - mu_hat[te_ind,i] - (X[te_ind,0] - t_eval[i])*beta_hat[te_ind,i]) * den_fac / (h**2 * sigmaK_sq) 
                # Self-normalizing weights
                w = np.sum(kern((t_eval[i] - X[te_ind,0])/h) * den_fac) / (n * h)
                norm_w[i] = norm_w[i] + w
                condST_int[te_ind,i] = condST_den
        
        if self_norm:
            IPW_hat = IPW_hat / norm_w
        # Add up the IPW and RA components
        theta_hat = IPW_hat + beta_hat * condST_int
        theta_est = np.mean(theta_hat, axis=0, where=~np.isnan(theta_hat))
        
        # Estimate the variance of theta(t) using the square of the influence function
        var_est = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            var_est[:,i] = (IPW_hat[:,i] + (np.mean(beta_hat[:,i] * condST_int[:,i]) - theta_est[i]))**2 * (h**3)
        sd_est = np.sqrt(np.mean(var_est, axis=0)/(n*(h**3)))
    return theta_est, sd_est

