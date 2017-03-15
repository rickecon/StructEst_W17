'''
---------------------------------------------------------------------------
MACS 40200: Final Project
Contributors: Bobae Kang
              Dongping Zhang
Python Version: 3.5
Seed: None

This is a Python script for MACS 40200 Structural Estimation final project.
The project estimates healthcare expenditure of individuals using a Markov
regime-swtiching model.
---------------------------------------------------------------------------
'''


# import packages
import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import seaborn
import markov_functions as mf
from markov_estimation import *
import os
from markov_functions import data_moments, erg_dist_vec, model_moments, \
                        criterion, spell_prob


'''

---------------------------------------------------------------------------
1.5. Plots
---------------------------------------------------------------------------
'''
# data moments and model moments with GMM estimates
dmom_vs_mmom = True
if dmom_vs_mmom:
    '''
    -------------------------------------------------------------------- 
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved 
    weights     = weights constructed for the distribution using the
                  fact the weights is equal to the probability of the
                  data we want
    bins        = a numpy array indicating the bin intervals
    -------------------------------------------------------------------- 
    ''' 
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    months_ix = np.array((range(data.shape[1])))
    plt.plot(months_ix, bk_vec_data, label='b(k) data')
    plt.plot(months_ix, bk_vec_gmm, label='b(k) model', linestyle='--')
    plt.plot(months_ix, dk_vec_data, label='d(k) data')
    plt.plot(months_ix, dk_vec_gmm, label='d(k) model', linestyle='--')
    plt.xticks(months_ix)
    plt.title('Compare data and model moments with GMM estimation', \
              fontsize=15)
    plt.xlabel('Number of consecutive months spells', fontsize=13)
    plt.ylabel('Percent of spells', fontsize=13)
    legend = plt.legend(loc='upper right', \
                        prop={'size':12}, \
                        handlelength=2, frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')

    # generate plot in the designated directory
    output_path = os.path.join(output_dir, 'Fig_1')
    plt.savefig(output_path)   
    plt.close()


# data moments and model moments with 2-step GMM estimate
dmom_vs_mmom_2step = True
if dmom_vs_mmom_2step:
    plt.plot(months_ix, bk_vec_data, label='b(k) data')
    plt.plot(months_ix, bk_vec_gmm2, label='b(k) model', linestyle='--')
    plt.plot(months_ix, dk_vec_data, label='d(k) data')
    plt.plot(months_ix, dk_vec_gmm2, label='d(k) model', linestyle='--')
    plt.xticks(months_ix)
    plt.xlim(0, 23.5)
    plt.title('Compare data and model moments with 2-step GMM estimation', \
              fontsize=15)
    plt.xlabel('Number of consecutive months spells', fontsize=13)
    plt.ylabel('Percent of spells', fontsize=13)
    legend = plt.legend(loc='upper right', prop={'size':12}, \
                        handlelength=2, frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')

    # generate plot in the designated directory
    output_path = os.path.join(output_dir, 'Fig_2')
    plt.savefig(output_path)   
    plt.close()


'''
---------------------------------------------------------------------------
Part 2. Esimating individual health care expenditures:
    2.1. Positive spell probabilities
    2.2. Data preparation for cost estimation
    2.3. Plots
---------------------------------------------------------------------------
2.1. Positive spell probabilities
---------------------------------------------------------------------------
'''
censored_none2, censored_none3, censored_none4 = \
    spell_prob(lambda_gmm2, delta_gmm2, data.shape[1], 'none')
censored_front2, censored_front3, censored_front4 = \
    spell_prob(lambda_gmm2, delta_gmm2, data.shape[1], 'front')
censored_back2, censored_back3, censored_back4 = \
    spell_prob(lambda_gmm2, delta_gmm2, data.shape[1], 'back')
censored_both2, censored_both3, censored_both4 = \
    spell_prob(lambda_gmm2, delta_gmm2, data.shape[1], 'both')

'''
---------------------------------------------------------------------------
2.2. Data preparation for cost estimation
---------------------------------------------------------------------------
'''

'''
---------------------------------------------------------------------------
2.3. Plots
---------------------------------------------------------------------------
'''
# uncensored spells
uncensored = True
if uncensored:
    plt.plot(months_ix[:-2], censored_none2, label='H2', marker='o')
    plt.plot(months_ix[:-2], censored_none3, label='H3', marker='v')
    plt.plot(months_ix[:-2], censored_none4, label='H4', marker='s')
    plt.xticks(months_ix)
    plt.xlim(-.5, 22.5)
    plt.ylim(-.05, 1.05)
    plt.title('Percent probability of uncensored positive strings by type', fontsize=13)
    plt.xlabel('Consecutive period spells', fontsize=12)
    plt.ylabel('Percent of total probability', fontsize=12)
    legend = plt.legend(loc='upper right', prop={'size':12}, handlelength=1.5, frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    # generate plot in the designated directory
    output_path = os.path.join(output_dir, 'Fig_3')
    plt.savefig(output_path)   
    plt.close()


# front-censored spells
front_censored = True
if front_censored:
    plt.plot(months_ix[:-1], censored_front2, label='H2', marker='o')
    plt.plot(months_ix[:-1], censored_front3, label='H3', marker='v')
    plt.plot(months_ix[:-1], censored_front4, label='H4', marker='s')
    plt.xticks(months_ix)
    plt.xlim(-.5, 23.5)
    plt.ylim(-.05, 1.05)
    plt.title('Percent probability of front-censored positive strings by type', fontsize=13)
    plt.xlabel('Consecutive period spells', fontsize=12)
    plt.ylabel('Percent of total probability', fontsize=12)
    legend = plt.legend(loc='upper right', prop={'size':12}, handlelength=1.5, frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    # generate plot in the designated directory
    output_path = os.path.join(output_dir, 'Fig_4')
    plt.savefig(output_path)   
    plt.close()

# back-censored spells
back_censored = True
if back_censored:
    plt.plot(months_ix[:-1], censored_back2, label='H2', marker='o')
    plt.plot(months_ix[:-1], censored_back3, label='H3', marker='v')
    plt.plot(months_ix[:-1], censored_back4, label='H4', marker='s')
    plt.xticks(months_ix)
    plt.xlim(-.5, 23.5)
    plt.ylim(-.05, 1.05)
    plt.title('Percent probability of back-censored positive strings by type', fontsize=13)
    plt.xlabel('Consecutive period spells', fontsize=12)
    plt.ylabel('Percent of total probability', fontsize=12)
    legend = plt.legend(loc='upper right', prop={'size':12}, handlelength=1.5, frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    # generate plot in the designated directory
    output_path = os.path.join(output_dir, 'Fig_5')
    plt.savefig(output_path)   
    plt.close()

# front- and back-censored spells
both_censored = True
if both_censored:
    plt.plot(months_ix, censored_both2, label='H2', marker='o')
    plt.plot(months_ix, censored_both3, label='H3', marker='v')
    plt.plot(months_ix, censored_both4, label='H4', marker='s')
    plt.xticks(months_ix)
    plt.xlim(-.5, 24.5)
    plt.ylim(-.05, 1.05)
    plt.title('Percent probability of front- and back-censored positive strings by type', fontsize=13)
    plt.xlabel('Consecutive period spells', fontsize=12)
    plt.ylabel('Percent of total probability', fontsize=12)
    legend = plt.legend(loc='center right', prop={'size':12}, handlelength=1.5, frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    # generate plot in the designated directory
    output_path = os.path.join(output_dir, 'Fig_6')
    plt.savefig(output_path)   
    plt.close()
