# Nonparametric Doubly Robust Inference on Dose-Response Curve and its Derivative
This repository contains Python3 code for implementing the regression adjustment (RA), inverse probability weighting (IPW), and doubly robust (DR) estimators of dose-response curve and its derivative with and without the positivity condition.

- Paper Reference: Nonparametric Doubly Robust Inference on Derivative of Dose-Response Curve: With and Without Positivity (2025+)

### Requirements

- Python >= 3.10 (earlier version might be applicable).
- [NumPy](http://www.numpy.org/), [scikit-learn](https://scikit-learn.org/stable/), [PyTorch](https://pytorch.org/) (for neural network models and auto-differentiation, [SciPy](https://www.scipy.org/) (for some [statistical functions](https://docs.scipy.org/doc/scipy/reference/stats.html)), and [Matplotlib](https://matplotlib.org/) (for plotting).
- Optional if only use the main functions: [pandas](https://pandas.pydata.org/) and [pickle](https://docs.python.org/3/library/pickle.html).

### File Descriptions

Some high-level descriptions of our Python scripts are as follows:

- **npDoseResponseDerivDR.py (Key file)**: This file contains the implementations of the RA, IPW, and DR estimators of the derivative of a dose-response curve with and without the positivity condition.
- **npDoseResponseDR.py (Key file)**: This file contains the implementations of the RA, IPW, and DR estimators of the dose-response curve under the positivity condition.
- **rbf.py (Key auxiliary file)**: This file contains the implementations of common kernel functions.
- **utils1.py (Key auxiliary file)**: This file contains the utility functions for the main functions for implementing our proposed methods.
- **Case Study -- Job Corps program (Final).ipynb**: This Jupyter Notebook contains detailed code for applying our proposed DR estimators to the Job Corps data, replicating the finite-difference method proposed by Colangelo and Lee (2020), and creating the comparative plot. (Reproducing Figure 4 in the arxiv version of our paper.)
- **Job_Corps_Data_Final.py**: This file contains the code of applying our proposed DR estimators to the Job Corps data (for parallel slurm jobs).
- **Plotting for Simulation 1 Without Positivity (L=1 or L=5).ipynb**: These two Jupyter Notebooks contain the code for plotting the simulation results when the data model violates the positivity condition. (Reproducing Figures 5 and 6 in the arxiv version of our paper.)
- **Plotting for Simulation 2 (L=1 or L=5).ipynb**: These two Jupyter Notebooks contain the code for plotting the simulation results when the data model satisfies the positivity condition. (Reproducing Figures 2 and 3 in the arxiv version of our paper.)
- **Sim_Nopos1.py** and **Sim_Nopos1_inner.py**: These two files contain the code for applying our (bias-corrected) IPW and DR estimators when the data model violates the positivity condition (for parallel slurm jobs).
- **Simulation2_nonsep_Replicate.py** and **Simulation2_self_norm.py**: These two files contain the code for applying our proposed estimators and replicating the finite-difference method proposed by Colangelo and Lee (2020) under the data model that satisfies the positivity condition (for parallel slurm jobs).
- **Synthesize_Nopos_Res.py**, **Synthesize_Res2_Replicate.py**, and **Synthesize_Res2_selfnorm.py**: These files are used to synthesize the outputs from the parallel slurm jobs.


### 1. Problem Setup
The dose-response curve is defined as <img src="https://latex.codecogs.com/svg.latex?&space;t\mapsto\,m(t)=\mathbb{E}\left[Y(t)\right]"/>, where <img src="https://latex.codecogs.com/svg.latex?&space;Y(t)"/> is the potential outcome that would have been observed under treatment level <img src="https://latex.codecogs.com/svg.latex?&space;T=t"/>. Accordingly, the derivative function of the dose-response curve is defined as <img src="https://latex.codecogs.com/svg.latex?&space;t\mapsto\theta(t)=\frac{d}{dt}\mathbb{E}\left[Y(t)\right]"/>.

### 2. Nonparametric Inference Under the Positivity Condition

Under some regularity conditions and the positivity condition (see Section 2 of our paper), the dose-response curve and its derivative can be identified as <img src="https://latex.codecogs.com/svg.latex?&space;m(t)=\mathbb{E}\left[\mu(t,\mathbf{S})\right]"/> and <img src="https://latex.codecogs.com/svg.latex?&space;\theta(t)=\mathbb{E}\left[\frac{\partial}{\partial\,t}\mu(t,\mathbf{S})\right]"/>, where <img src="https://latex.codecogs.com/svg.latex?&space;\mu(t,\mathbf{s})=\mathbb{E}\left(Y|T=t,\mathbf{S}=\mathbf{s}\right)"/> is the conditional mean outcome (or regression) function of the outcome variable <img src="https://latex.codecogs.com/svg.latex?&space;Y\in\mathcal{Y}\subset\mathbb{R}"/> given the treatment <img src="https://latex.codecogs.com/svg.latex?&space;T=t\in\mathcal{T}\subset\mathbb{R}"/> and the covariate vector <img src="https://latex.codecogs.com/svg.latex?&space;\mathbf{S}=\mathbf{s}\in\mathcal{S}\subset\mathbb{R}^d"/>.

For nonparametric estimation of the dose-response curve <img src="https://latex.codecogs.com/svg.latex?&space;t\mapsto\,m(t)=\mathbb{E}\left[Y(t)\right]"/> with observed data <img src="https://latex.codecogs.com/svg.latex?&space;\left\{(Y_i,T_i,\mathbf{S}_i)\right\}_{i=1}^n"/>, there are three major strategies:

* **Regression Adjustment (RA) Estimator:**
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;\widehat{m}_{\mathrm{RA}}(t)=\frac{1}{n}\sum_{i=1}^n\widehat{\mu}(t,\mathbf{S}_i),"/>
</p>

where <img src="https://latex.codecogs.com/svg.latex?&space;\widehat{\mu}(t,\mathbf{s})"/> is a (consistent) estimator of the conditional mean outcome function <img src="https://latex.codecogs.com/svg.latex?&space;\mu(t,\mathbf{s})"/>.

* **Inverse Probability Weighting (IPW) Estimator:**
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\widehat{m}_{\mathrm{IPW}}(t)=\frac{1}{nh}\sum_{i=1}^n\frac{Y_i\cdot\,K\left(\frac{T_i-t}{h}\right)}{\widehat{p}_{T|\mathbf{S}}(T_i|\mathbf{S}_i)},"/>
</p>

where <img src="https://latex.codecogs.com/svg.latex?&space;h>0"/> is a smoothing bandwidth parameter and <img src="https://latex.codecogs.com/svg.latex?&space;\widehat{p}_{T|\mathbf{S}}(t|\mathbf{s})"/> is a (consistent) estimator of the conditional density <img src="https://latex.codecogs.com/svg.latex?&space;p_{T|\mathbf{S}}(t|\mathbf{s})"/>.

* **Doubly Robust (DR) Estimator:**
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\widehat{m}_{\mathrm{DR}}(t)=\frac{1}{nh}\sum_{i=1}^n\left\{\frac{K\left(\frac{T_i-t}{h}\right)}{\widehat{p}_{T|\mathbf{S}}(T_i|\mathbf{S}_i)}\cdot\left[Y_i-\widehat{\mu}(t,\mathbf{S}_i)\right]+h\cdot\widehat{\mu}(t,\mathbf{S}_i)\right\},"/>
</p>

where <img src="https://latex.codecogs.com/svg.latex?&space;\widehat{\mu}(t,\mathbf{s})"/> and <img src="https://latex.codecogs.com/svg.latex?&space;\widehat{p}_{T|\mathbf{S}}(t|\mathbf{s})"/> are (consistent) estimators of <img src="https://latex.codecogs.com/svg.latex?&space;\mu(t,\mathbf{s})"/> and <img src="https://latex.codecogs.com/svg.latex?&space;p_{T|\mathbf{S}}(t|\mathbf{s})"/>, respectively. The DR estimator is not only robust to the misspecifications of the conditional mean outcome and conditional density models but also asymptotically normal as
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\sqrt{nh}\left[\widehat{m}_{\mathrm{DR}}(t)-m(t)-h^2B_m(t)\right]\stackrel{d}{\to}\mathcal{N}\left(0,V_m(t)\right),"/>
</p>

where <img src="https://latex.codecogs.com/svg.latex?&space;h^2B_m(t)"/> is the bias term that shrinks quadratically to 0 with respect to <img src="https://latex.codecogs.com/svg.latex?&space;h>0"/> and <img src="https://latex.codecogs.com/svg.latex?&space;V_m(t)"/> is the asymptotic variance term that can be estimated by
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\widehat{V}_m(t)=\frac{1}{n}\sum_{i=1}^n\left\{\frac{K\left(\frac{T_i-t}{h}\right)}{\sqrt{h}\cdot\widehat{p}_{T|\mathbf{S}}(T_i|\mathbf{S}_i)}\left[Y_i-\widehat{\mu}(t,\mathbf{S}_i)\right]+\sqrt{h}\left[\widehat{\mu}(t,\mathbf{S}_i)-\widehat{m}_{\mathrm{DR}}(t)\right]\right\}^2."/>
</p>

More details can be found in Section 2.1 and Section D of our paper.

For nonparametric estimation of the derivative function <img src="https://latex.codecogs.com/svg.latex?&space;t\mapsto\theta(t)=\frac{d}{dt}\mathbb{E}\left[Y(t)\right]"/>, we also consider three similar strategies:

* **Regression Adjustment (RA) Estimator:**
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;\widehat{\theta}_{\mathrm{RA}}(t)=\frac{1}{n}\sum_{i=1}^n\widehat{\beta}(t,\mathbf{S}_i),"/>
</p>

where <img src="https://latex.codecogs.com/svg.latex?&space;\widehat{\beta}(t,\mathbf{s})"/> is a (consistent) estimator of <img src="https://latex.codecogs.com/svg.latex?&space;\beta(t,\mathbf{s})=\frac{\partial}{\partial\,t}\mu(t,\mathbf{s})"/>.

* **Inverse Probability Weighting (IPW) Estimator:**
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\widehat{\theta}_{\mathrm{IPW}}(t)=\frac{1}{nh^2}\sum_{i=1}^n\frac{Y_i\left(\frac{T_i-t}{h}\right)K\left(\frac{T_i-t}{h}\right)}{\kappa_2\cdot\widehat{p}_{T|\mathbf{S}}(T_i|\mathbf{S}_i)},"/>
</p>

where <img src="https://latex.codecogs.com/svg.latex?&space;K:\mathbb{R}\to[0,\infty)"/> is a kernel function with <img src="https://latex.codecogs.com/svg.latex?&space;\kappa_2=\int\,u^2K(u)\,du"/>, <img src="https://latex.codecogs.com/svg.latex?&space;h>0"/> is a smoothing bandwidth parameter and <img src="https://latex.codecogs.com/svg.latex?&space;\widehat{p}_{T|\mathbf{S}}(t|\mathbf{s})"/> is a (consistent) estimator of the conditional density <img src="https://latex.codecogs.com/svg.latex?&space;p_{T|\mathbf{S}}(t|\mathbf{s})"/>.


* **Doubly Robust (DR) Estimator:**
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\widehat{\theta}_{\mathrm{DR}}(t)=\frac{1}{nh}\sum_{i=1}^n\left\{\frac{\left(\frac{T_i-t}{h}\right)K\left(\frac{T_i-t}{h}\right)}{h\cdot\kappa_2\cdot\widehat{p}_{T|\mathbf{S}}(T_i|\mathbf{S}_i)}\cdot\left[Y_i-\widehat{\mu}(t,\mathbf{S}_i)-\left(T_i-t\right)\widehat{\beta}(t,\mathbf{S}_i)\right]+h\cdot\widehat{\beta}(t,\mathbf{S}_i)\right\},"/>
</p>

where <img src="https://latex.codecogs.com/svg.latex?&space;\widehat{\mu}(t,\mathbf{s}),\,\widehat{\beta}(t,\mathbf{s})"/> and <img src="https://latex.codecogs.com/svg.latex?&space;\widehat{p}_{T|\mathbf{S}}(t|\mathbf{s})"/> are (consistent) estimators of <img src="https://latex.codecogs.com/svg.latex?&space;\mu(t,\mathbf{s}),\,\beta(t,\mathbf{s})"/> and <img src="https://latex.codecogs.com/svg.latex?&space;p_{T|\mathbf{S}}(t|\mathbf{s})"/>, respectively. Again, our proposed DR estimator of <img src="https://latex.codecogs.com/svg.latex?&space;\theta(t)"/> is not only robust to the misspecifications of the conditional mean outcome and conditional density models but also asymptotically normal as
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\sqrt{nh^3}\left[\widehat{\theta}_{\mathrm{DR}}(t)-\theta(t)-h^2B_{\theta}(t)\right]\stackrel{d}{\to}\mathcal{N}\left(0,V_{\theta}(t)\right),"/>
</p>

where <img src="https://latex.codecogs.com/svg.latex?&space;h^2B_{\theta}(t)"/> is the bias term that shrinks quadratically to 0 with respect to <img src="https://latex.codecogs.com/svg.latex?&space;h>0"/> and <img src="https://latex.codecogs.com/svg.latex?&space;V_{\theta}(t)"/> is the asymptotic variance term that can be estimated by
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\widehat{V}_{\theta}(t)=\frac{1}{n}\sum_{i=1}^n\left\{\frac{\left(\frac{T_i-t}{h}\right)K\left(\frac{T_i-t}{h}\right)}{\sqrt{h}\cdot\kappa_2\cdot\widehat{p}_{T|\mathbf{S}}(T_i|\mathbf{S}_i)}\left[Y_i-\widehat{\mu}(t,\mathbf{S}_i)-(T_i-t)\widehat{\beta}(t,\mathbf{S}_i)\right]+\sqrt{h^3}\left[\widehat{\beta}(t,\mathbf{S}_i)-\widehat{\theta}_{\mathrm{DR}}(t)\right]\right\}^2."/>
</p>

More details can be found in Section 3 of our paper.

### 3. Nonparametric Inference Without the Positivity Condition

Both the dose-response curve <img src="https://latex.codecogs.com/svg.latex?&space;m(t)=\mathbb{E}\left[Y(t)\right]"/> and its derivative <img src="https://latex.codecogs.com/svg.latex?&space;\theta(t)=\frac{d}{dt}\mathbb{E}\left[Y(t)\right]"/> are unidentifiable in general when the positivity condition is violated; see Section 4.1 in our paper. To identify and estimate them, we assume an additive structure on the potential outcome <img src="https://latex.codecogs.com/svg.latex?&space;Y(t)=\bar{m}(t)+\eta(\mathbf{S})+\epsilon"/> so that
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\theta(t)=\bar{m}'(t)=\mathbb{E}\left[\frac{\partial}{\partial\,t}\mu(t,\mathbf{S})\Big|T=t\right],\quad\,m(t)=\mathbb{E}\left[Y+\int_T^t\theta(\tilde{t})\,d\tilde{t}\right]=\mathbb{E}\left\{Y+\int_T^t\mathbb{E}\left[\frac{\partial}{\partial\,t}\mu(T,\mathbf{S})\Big|T=\tilde{t}\right]\,d\tilde{t}\right\}."/>
</p>

These formulas lead to the **RA estimator** of <img src="https://latex.codecogs.com/svg.latex?&space;\theta(t)"/> as:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;\widehat{\theta}_{\mathrm{C,RA}}(t)=\int\widehat{\beta}(t,\mathbf{s})\,d\hat{F}_{\mathbf{S}|T}(\mathbf{s}|t)."/>
</p>

The above IPW and DR estimators will give rise to inconsistent estimates of <img src="https://latex.codecogs.com/svg.latex?&space;\theta(t)"/> without the positivity even when the additive model holds true. We proposed our bias-corrected **IPW and DR estimators** of <img src="https://latex.codecogs.com/svg.latex?&space;\theta(t)"/> as:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;\widehat{\theta}_{\mathrm{C,IPW}}(t)=\frac{1}{nh^2}\sum_{i=1}^n\frac{Y_i\left(\frac{T_i-t}{h}\right)K\left(\frac{T_i-t}{h}\right)\widehat{p}_{\zeta}(\mathbf{S}_i|t)}{\kappa_2\cdot\widehat{p}(T_i|\mathbf{S}_i)},"/>

<img src="https://latex.codecogs.com/svg.latex?\large&space;\widehat{\theta}_{\mathrm{C,DR}}(t)=\frac{1}{nh^2}\sum_{i=1}^n\frac{\left(\frac{T_i-t}{h}\right)K\left(\frac{T_i-t}{h}\right)\widehat{p}_{\zeta}(\mathbf{S}_i|t)}{\kappa_2\cdot\widehat{p}(T_i|\mathbf{S}_i)}\left[Y_i-\widehat{\mu}(t,\mathbf{S}_i)-(T_i-t)\widehat{\beta}(t,\mathbf{S}_i)\right]+\int\widehat{\beta}(t,\mathbf{s})\cdot\widehat{p}_{\zeta}(\mathbf{s}|t)\,d\mathbf{s},"/>
</p>

where <img src="https://latex.codecogs.com/svg.latex?&space;\widehat{\mu}(t,\mathbf{s}),\,\widehat{\beta}(t,\mathbf{s})"/>, and <img src="https://latex.codecogs.com/svg.latex?&space;\widehat{p}(t,\mathbf{s}),\widehat{p}_{\zeta}(\mathbf{s}|t)"/> are (consistent) estimators of <img src="https://latex.codecogs.com/svg.latex?&space;\mu(t,\mathbf{s}),\,\beta(t,\mathbf{s})"/>, the joint density <img src="https://latex.codecogs.com/svg.latex?&space;p(t|\mathbf{s})"/>, and the interior conditional density <img src="https://latex.codecogs.com/svg.latex?&space;p_{\zeta}(\mathbf{s}|t)"/> respectively. We also prove that this bias-corrected DR estimator is not only robust to the misspecifications of the conditional mean outcome and conditional density models but also asymptotically normal as
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\sqrt{nh^3}\left[\widehat{\theta}_{\mathrm{C,DR}}(t)-\theta(t)-h^2B_{C,\theta}(t)\right]\stackrel{d}{\to}\mathcal{N}\left(0,V_{C,\theta}(t)\right)."/>
</p>

More details can be found in Section 5 of our paper.

### 3. Example Code

#### Estimation of Dose-Response Curve and its Derivative Under Positivity

```bash
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from npDoseResponseDR import DRCurve
from npDoseResponseDerivDR import DRDerivCurve, NeurNet
from sklearn.neural_network import MLPRegressor

rho = 0.5  # correlation between adjacent Xs
d = 20   # Dimension of the confounding variables
n = 2000

Sigma = np.zeros((d,d)) + np.eye(d)
for i in range(d):
    for j in range(i+1, d):
        if (j < i+2) or (j > i+d-2):
            Sigma[i,j] = rho
            Sigma[j,i] = rho
sig = 1

np.random.seed(123)
# Data generating process
X_sim = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
nu = np.random.randn(n)
eps = np.random.randn(n)
theta = 1/(np.linspace(1, d, d)**2)

T_sim = scipy.stats.norm.cdf(3*np.dot(X_sim, theta)) + 3*nu/4 - 1/2
Y_sim = 1.2*T_sim + T_sim**2 + T_sim*X_sim[:,0] + 1.2*np.dot(X_sim, theta) + eps*np.sqrt(0.5+ scipy.stats.norm.cdf(X_sim[:,0]))
X_dat = np.column_stack([T_sim, X_sim])
t_qry = np.linspace(-2, 2, 41)

# Choice of the bandwidth parameter
h = 4*np.std(T_sim)*n**(-1/5)

# RA estimator of m(t)
reg_mod = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', learning_rate='adaptive', 
                       learning_rate_init=0.1, random_state=1, max_iter=200)
m_est_ra5 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="RA", mu=reg_mod, 
                    L=5, h=None, kern="epanechnikov", print_bw=False)

# IPW estimator of m(t)
regr_nn2 = MLPRegressor(hidden_layer_sizes=(20,), activation='relu', learning_rate='adaptive', 
                        learning_rate_init=0.1, random_state=1, max_iter=200)
m_est_ipw5 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="IPW", mu=None, 
                     condTS_type='kde', condTS_mod=regr_nn2, tau=0.001, L=5, h=h, 
                     kern="epanechnikov", h_cond=None, self_norm=True, print_bw=True)

# DR estimator of m(t)
m_est_dr5, sd_est_dr5 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="DR", mu=reg_mod, 
                                condTS_type='kde', condTS_mod=regr_nn2, tau=0.001, L=5, 
                                h=h, kern="epanechnikov", h_cond=None, self_norm=True, print_bw=True)

# RA estimator of \theta(t)
theta_ra5 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="RA", beta_mod=NeurNet, 
                         n_iter=1000, lr=0.01, L=5, print_bw=False)

# IPW estimator of \theta(t)
theta_ipw5 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="IPW", beta_mod=None, 
                          condTS_type='kde', condTS_mod=regr_nn2, tau=0.001, L=5, 
                          h=h, kern="epanechnikov", h_cond=None, self_norm=True, print_bw=True)

# DR estimator of \theta(t)
theta_dr5, theta_sd5 = DRDerivCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="DR", beta_mod=NeurNet, 
                                    n_iter=1000, lr=0.01, condTS_type='kde', condTS_mod=regr_nn2, tau=0.001, 
                                    L=5, h=h, kern="epanechnikov", h_cond=None, self_norm=True, print_bw=True)

## Visualization of the estimators
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(t_qry, 1.2*t_qry + t_qry**2, linewidth=4, color='red', label=r'True $m(t)$')
plt.plot(t_qry, m_est_ra5, linewidth=2, color='darkorange', label=r'$\widehat{m}_{\mathrm{RA}}(t)$')
plt.plot(t_qry, m_est_ipw5, linewidth=2, alpha=0.8, color='darkgreen', label=r'$\widehat{m}_{\mathrm{IPW}}(t)$')
plt.plot(t_qry, m_est_dr5, linewidth=2, alpha=0.8, color='blue', label=r'$\widehat{m}_{\mathrm{DR}}(t)$')
plt.fill_between(t_qry, m_est_dr5 - sd_est_dr5*scipy.stats.norm.ppf(0.975), 
                 m_est_dr5 + sd_est_dr5*scipy.stats.norm.ppf(0.975), 
                 color='blue', alpha=.3, label='95% pointwise CIs')
plt.legend(fontsize=13)
plt.xlabel('Treatment value t')
plt.ylabel(r'Dose-response curve $m(t)$ or $\widehat{m}(t)$')

plt.subplot(1, 2, 2)
plt.plot(t_qry, 1.2 + 2*t_qry, linewidth=4, color='red', label=r'True $\theta(t)$')
plt.plot(t_qry, theta_ra5, linewidth=2, color='darkorange', label=r'$\widehat{\theta}_{\mathrm{RA}}(t)$')
plt.plot(t_qry, theta_ipw5, linewidth=2, alpha=0.8, color='darkgreen', label=r'$\widehat{\theta}_{\mathrm{IPW}}(t)$')
plt.plot(t_qry, theta_dr5, linewidth=2, alpha=0.8, color='blue', label=r'$\widehat{\theta}_{\mathrm{DR}}(t)$')
plt.fill_between(t_qry, theta_dr5 - theta_sd5*scipy.stats.norm.ppf(0.975), 
                 theta_dr5 + theta_sd5*scipy.stats.norm.ppf(0.975), 
                 color='blue', alpha=.3, label='95% pointwise CIs')
plt.legend(fontsize=13)
plt.xlabel('Treatment value t')
plt.ylabel(r'Derivative curve $\theta(t)$ or $\widehat{\theta}(t)$')

plt.tight_layout()
plt.savefig('./Figures/est_illust_pos.png')
```

<p align="center">
<img src="https://github.com/zhangyk8/npDRDeriv/blob/main/Figures/est_illust_pos.png" style="zoom:60%" />
 <br><B>Fig 1. </B>Illustrative plots of RA, IPW, and DR estimators of a dose-response curve and its derivative under positivity.
</p>


#### Dose-Response Curve Derivative Estimation Under Positivity

```bash
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from npDoseResponseDerivDR import NeurNet, RADRDerivBC, IPWDRDerivBC, DRDRDerivBC

n = 2000
np.random.seed(123)
# Data generating process
X_sim = 2*np.random.rand(n) - 1
T_sim = np.sin(np.pi*X_sim) + np.random.rand(n)*0.6 - 0.3
Y_sim = T_sim**2 + T_sim**3 + 10*X_sim + np.random.normal(loc=0, scale=1, size=n)
X_dat = np.concatenate([T_sim.reshape(-1,1), X_sim.reshape(-1,1)], axis=1)

# Query points 
t_qry = np.linspace(-0.8, 0.8, 41)

# Choice of the bandwidth parameter
h = 2*np.std(T_sim)*n**(-1/5)

# Bias-corrected RA estimator of \theta(t)
theta_C_RA5 = RADRDerivBC(Y=Y_sim, X=X_dat, t_eval=t_qry, mu=NeurNet, L=5, 
                          n_iter=1000, lr=0.01, h_bar=None, kernT_bar="gaussian")

# Bias-corrected IPW estimator of \theta(t)
theta_C_IPW5, condTS5 = IPWDRDerivBC(Y=Y_sim, X=X_dat, t_eval=t_qry, L=5, h=h, kern='epanechnikov', 
                                     b=None, self_norm=True, thres_val=0.85)

# Bias-corrected DR estimator of \theta(t)
theta_C_DR5, theta_C_sd5 = DRDRDerivBC(Y=Y_sim, X=X_dat, t_eval=t_qry, mu=NeurNet, L=5, h=h, kern='epanechnikov', 
                                       n_iter=1000, lr=0.01, b=None, thres_val=0.85, self_norm=False)

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(6,5))
plt.plot(t_qry, 2*t_qry + 3*(t_qry**2), linewidth=4, color='red', label=r'True $\theta(t)$')
plt.plot(t_qry, theta_C_RA5, linewidth=2, alpha=0.8, color='darkorange', label=r'$\widehat{\theta}_{\mathrm{C,RA}}(t)$')
plt.plot(t_qry, theta_C_IPW5, linewidth=2, alpha=0.8, color='darkgreen', label=r'$\widehat{\theta}_{\mathrm{C,IPW}}(t)$')
plt.plot(t_qry, theta_C_DR5, linewidth=2, alpha=0.8, color='blue', label=r'$\widehat{\theta}_{\mathrm{C,DR}}(t)$')
plt.fill_between(t_qry, theta_C_DR5 - theta_C_sd5*scipy.stats.norm.ppf(0.975), 
                 theta_C_DR5 + theta_C_sd5*scipy.stats.norm.ppf(0.975), 
                 color='blue', alpha=.3, label='95% pointwise CIs')
plt.legend()
plt.xlabel('Treatment value t')
plt.ylabel(r'Derivative curve $\theta(t)$ or $\widehat{\theta}_C(t)$')
plt.tight_layout()
plt.savefig('./Figures/est_illust_nopos.png')
```

<p align="center">
<img src="https://github.com/zhangyk8/npDRDeriv/blob/main/Figures/est_illust_nopos.png" style="zoom:60%" />
 <br><B>Fig 2. </B>Illustrative plot of bias-corrected RA, IPW, and DR estimators of the derivative of a dose-response curve without positivity.
</p>

### Additional References

- K. Colangelo and Y.-Y. Lee (2020). Double debiased machine learning nonparametric inference with continuous treatments. _arXiv preprint arXiv:2004.03036_.
