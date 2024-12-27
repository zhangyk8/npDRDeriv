# Nonparametric Doubly Robust Inference on Dose-Response Curve and its Derivative
This repository contains Python3 code for implementing the regression adjustment (RA), inverse probability weighting (IPW), and doubly robust (DR) estimators of dose-response curve and its derivative with and without the positivity condition.

- Paper Reference: Nonparametric Doubly Robust Inference on Derivative of Dose-Response Curve: With and Without Positivity (2025+)

### Requirements

- Python >= 3.10 (earlier version might be applicable).
- [NumPy](http://www.numpy.org/), [scikit-learn](https://scikit-learn.org/stable/), [PyTorch](https://pytorch.org/) (for neural network models and auto-differentiation, [SciPy](https://www.scipy.org/) (for some [statistical functions](https://docs.scipy.org/doc/scipy/reference/stats.html)), and [Matplotlib](https://matplotlib.org/) (for plotting).
- Optional if only use the main functions: [pandas](https://pandas.pydata.org/) and [pickle](https://docs.python.org/3/library/pickle.html).

### Descriptions

Some high-level descriptions of our Python scripts are as follows:

- **Case Study -- Job Corps program (Final).ipynb**: This Jupyter Notebook contains detailed code for applying our proposed DR estimators to the Job Corps data, replicating the method proposed by Colangelo and Lee (2020), and creating the comparative plot. (Reproducing Figure 4 in the arxiv version of our paper.)
- **Job_Corps_Data_Final.py**: The file contains the code of applying our proposed DR estimators to the Job Corps data (for parallel slurm jobs).
- 
