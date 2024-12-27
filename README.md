# Nonparametric Doubly Robust Inference on Dose-Response Curve and its Derivative
This repository contains Python3 code for implementing the regression adjustment (RA), inverse probability weighting (IPW), and doubly robust (DR) estimators of dose-response curve and its derivative with and without the positivity condition.

- Paper Reference: Nonparametric Doubly Robust Inference on Derivative of Dose-Response Curve: With and Without Positivity (2025+)

### Requirements

- Python >= 3.10 (earlier version might be applicable).
- [NumPy](http://www.numpy.org/), [scikit-learn](https://scikit-learn.org/stable/), [PyTorch](https://pytorch.org/) (for neural network models and auto-differentiation, [SciPy](https://www.scipy.org/) (for some [statistical functions](https://docs.scipy.org/doc/scipy/reference/stats.html)), and [Matplotlib](https://matplotlib.org/) (for plotting).
- Optional if only use the main functions: [pandas](https://pandas.pydata.org/) and [pickle](https://docs.python.org/3/library/pickle.html).

### Descriptions

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
