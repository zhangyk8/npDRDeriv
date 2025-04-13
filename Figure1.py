#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: April 4, 2025

Description: This script creates the illustrative plot of derivative effect curves
(Figure 1 in the paper).
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

if __name__ == "__main__":
    t = np.linspace(1, 6, 200)
    m_t = scipy.stats.norm.pdf(t, 3.5, 0.7)

    def m_deriv(x, mu=0, sig=1):
        val1 = (mu - x)/(np.sqrt(2*np.pi)*(sig)**3)
        val2 = np.exp(-(x - mu)**2/(2*sig**2))
        return val1 * val2

    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(t, m_t, color='red', linewidth=4, label=r'$m(t)=\mathbb{E}[Y(t)]$')
    plt.axhline(y=0.25, linestyle='dashed', color='darkorange', linewidth=3)
    plt.plot([2.6], [0.25], 'o', color='black', markersize=7)
    plt.plot([4.398], [0.25], 'o', color='black', markersize=7) 
    # plt.ylim([0, 0.46])
    # plt.xlim([0, 6])
    plt.text(1.7, 0.27, r"$m(t_1)$")
    plt.text(4.46, 0.27, r"$m(t_2)$")
    plt.legend(loc='lower center', fontsize=13, bbox_to_anchor=(0.5, -0.02))
    plt.xlabel(r'Treatment value $T=t$')
    plt.ylabel(r'$m(t)$')

    plt.subplot(122)
    plt.plot(t, m_deriv(t, 3.5, 0.7), color='cyan', linewidth=4, label=r'$\theta(t)=\frac{d}{dt}\mathbb{E}[Y(t)]$')
    plt.axhline(y=0, linestyle='dotted', color='tab:purple', linewidth=4, label=r'$\mathbb{E}\left[\theta(T)\right]$')
    # plt.axvline(x=2.6, linestyle='dashed', color='grey', linewidth=2)
    # plt.axvline(x=4.398, linestyle='dashed', color='grey', linewidth=2)
    plt.plot([2.6], [0.45], 'o', color='black', markersize=7)
    plt.plot([4.398], [-0.45], 'o', color='black', markersize=7) 
    plt.text(1.7, 0.43, r"$\theta(t_1)$")
    plt.text(4.56, -0.49, r"$\theta(t_2)$")
    plt.legend(loc='lower left', bbox_to_anchor=(-0.015, -0.02), fontsize=13)
    plt.xlabel(r'Treatment value $T=t$')
    plt.ylabel(r'$\theta(t)$')

    plt.tight_layout()
    plt.savefig('./Figures/dr_deriv.pdf')