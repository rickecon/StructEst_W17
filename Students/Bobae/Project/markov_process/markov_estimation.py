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

'''
---------------------------------------------------------------------------
Part 0. Preparation:
    0.1. Import required packages for analysis
    0.2. Import raw data and data cleaning
---------------------------------------------------------------------------
claim_data  = a pandas (100000, 27) dataframe of health expenditure
non_zeros   = a boolean vector of length 100000 and would return a false
              if any row has a negative health expenditure
data_nonneg = a new pandas dataframe (99965, 27) of health expenditure
              after removing 35 rows that contains negative health expenditrue
exp         = a (99965, 24) dataframe that contins only the monthly expenditure
              columns of the dataset
exp_arr     = a (99965, 24) numpy array verion of exp
exp_train   = a (99965, 12) pandas dataframe that contains the first half of
              exp and would be treated as the training data
exp_test    = a (99965, 12) pandas dataframe that contains the second half of
              exp and would be treated as the testing data
---------------------------------------------------------------------------
'''
# import packages
import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import seaborn
import warnings
warnings.filterwarnings('ignore')
import markov_functions as mf
from markov_functions import data_moments, erg_dist_vec, model_moments, \
                        criterion, spell_prob

# import the data
claim_data = pd.read_pickle('claims_adj.pkl')

# Attention:
# there are negative expenditures in the dataset: remove them
# 1. find all negative expenditure by booleanizing the dataframe
non_zeros = (claim_data.iloc[:,3:] < 0).sum(axis = 1) == 0
# 2. remove those rows (subjects) if the sum of those booleans != 0
data_nonneg = claim_data[non_zeros]

# extract monthly expenditures
exp = data_nonneg.iloc[:,3:]
# numpy array of the monthly expenditures to facilitate operations
exp_arr = np.array(exp)
# training set (first 12 months)
exp_train = exp.iloc[:,:12]
# test set (second 12 months)
exp_test = exp.iloc[:,12:]


'''
---------------------------------------------------------------------------
Part 1. Esimating the Markov process:
    1.1. GMM estimation
    1.2. Report GMM outcomes
    1.3. GMM estimation, 2-step
    1.4. Report GMM outcomes, 2-step
    1.5. Plots
---------------------------------------------------------------------------
1.1. GMM estimation
---------------------------------------------------------------------------
data         = the same object as exp, which is a (99965, 24) dataframe
               that contins only the monthly expenditure columns of the
               dataset
lambda0_init = the initial value of lambda0
lambda1_init = the initial value of lambda1
lambda2_init = the initial value of lambda2
lambda3_init = the initial value of lambda3
lambda4_init = the initial value of lambda4
lambda_init  = a numpy array of size (5,) that contains all lambdas0 - lambda4
delta1_init  = the initial value of delta1
delta2_init  = the initial value of delta2
delta3_init  = the initial value of delta3
dealta_init  = a numpy array of size (3,) that contains all delta1-delta3
params_init  = a numpy array of size (8,) that appends all lambdas and all
               deltas into one numpy array
months       = an integer of total number of time periods
W_hat        = the identity weighting matrix that would be used in the
               optimization procedure
bk_vec_data  = a numpy array of size (24,) and it is the data moment of bk
dk_vec_data  = a numpy array of size (24,) and it is the data moment of dk
gmm_args     = all arguments needed for the minimization/optimization
               algorithm, and they are:
                1. bk_vec_data
                2. dk_vec_data
                3. W_hat
                4. A boolean
cons         = a tuple of dictionaries in which each dictionary has two keys
                    - type : ineq
                    - fun  : contraint function
               which are the lower bound and the upper bound of the
               constrained parameters
bounds       = a tuple of tuples of length 8 in which each nested tuple
               represents the bounds of a parameter, which is in the interval
               of (0, 1)
results      = an minimization object that contains the result of the
               minimization information.
---------------------------------------------------------------------------
'''
# set what data to be used for GMM estimation
data = exp

# set initial parameter values
lambda0_init = 0.5
lambda1_init = 0.25
lambda2_init = 0.25
lambda3_init = 0.25
lambda4_init = 0.25
# vectorize lambda initial values
lambda_init = np.array([lambda0_init, lambda1_init, lambda2_init,
                        lambda3_init, lambda4_init])

delta1_init = (1 - lambda0_init) / 4
delta2_init = (1 - lambda0_init) / 4
delta3_init = (1 - lambda0_init) / 4
# vectorize delta initial values
delta_init = np.array([delta1_init, delta2_init, delta3_init])
params_init = np.append(lambda_init, delta_init)

# set argument values
months = data.shape[1]
W_hat = np.eye(2*months-1)
bk_vec_data, dk_vec_data = data_moments(data)
gmm_args = (bk_vec_data, dk_vec_data, W_hat, months, True)

# set constraints
cons = ({'type': 'ineq',
          'fun': lambda x: 1 - x[0] - x[5] - x[6] - x[7] - 1e-9}, # lower bound
        {'type': 'ineq',
          'fun': lambda x: x[0] + x[5] + x[6] + x[7] + 1e-9}) # upper bound

# set bounds
bounds = ((1e-10, 1 - 1e-10), (1e-10, 1 - 1e-10),
          (1e-10, 1 - 1e-10), (1e-10, 1 - 1e-10),
          (1e-10, 1 - 1e-10), (1e-10, 1 - 1e-10),
          (1e-10, 1 - 1e-10), (1e-10, 1 - 1e-10))


# run optimization
results = opt.minimize(criterion, params_init, args=(gmm_args), \
                       method='SLSQP', constraints=cons, \
                       bounds=bounds)


'''
---------------------------------------------------------------------------
1.2. Report GMM outcomes
---------------------------------------------------------------------------
lambda_gmm  = a numpy array of (5,) lambda0 to lambda4
lambda0_gmm = the gmm estimator of lambda0
lambda1_gmm = the gmm estimator of lambda1
lambda2_gmm = the gmm estimator of lambda2
lambda3_gmm = the gmm estimator of lambda3
lambda4_gmm = the gmm estimator of lambda4
delta1_gmm  = the gmm estimator of delta1
delta2_gmm  = the gmm estimator of delta2
delta3_gmm  = the gmm estimator of delta3
delta4_gmm  = the gmm estimator of delta4
delta_gmm   = a numpy array of (4.) delta1 to delta4
bk_vec_gmm  = the model moments bk using the gmm estimators
dk_vec_gmm  = the model moments dk using the gmm estimators
---------------------------------------------------------------------------
'''
# unpack and organize results
lambda_gmm = results.x[:5]
lambda0_gmm, lambda1_gmm, lambda2_gmm, lambda3_gmm, lambda4_gmm = lambda_gmm
delta1_gmm, delta2_gmm, delta3_gmm = results.x[5:]
delta4_gmm = 1 - lambda0_gmm - delta1_gmm - delta2_gmm - delta3_gmm
delta_gmm = np.array([delta1_gmm, delta2_gmm, delta3_gmm, delta4_gmm])
# get model moments
bk_vec_gmm, dk_vec_gmm = model_moments(lambda_gmm, delta_gmm[:-1], months)

# report GMM parameters
print('----------------------------------------------------------------------')
print('1-Step GMM Estimation')
print('----------------------------------------------------------------------')
print('Optimization Result')
print(results)
print('----------------------------------------------------------------------')
print('GMM Estimators: Lambdas')
print(lambda_gmm)
print('GMM Estimators: Deltas')
print(delta_gmm)
print('----------------------------------------------------------------------')
print('Data Moments vs. Model Moments')
print('----------------------------------------------------------------------')
print('bk data')
print(bk_vec_data)
print('bk_gmm')
print(bk_vec_gmm)
print('-------------------------------------')
print('dk_data')
print(dk_vec_data)
print('dk_gmm')
print(dk_vec_gmm)
print('----------------------------------------------------------------------')
print()
print()
print()
print()

'''
---------------------------------------------------------------------------
1.3. GMM estimation, 2-step
---------------------------------------------------------------------------
params_2step = a numpy array of size (8,) that appends all gmm estimators of
               lambdas and deltas into one numpy array
data_moms    = a numpy array of size (47,) that appends data moments of
               bk and dk
gmm_moms     = a numpy array of size (47,) that appends model moments of
               bk and dk
err2         = a numpy array of dimension (47, 1) which are the errors of
               gmm model moments from data moments
VCV2         = a numpy array of dimension (47, 47) which is the variance-
               covariance matrix of the 47 estimators
W_hat2       = the inverse of the vcv matrix
gmm_args2    = all arguments needed for the minimization/optimization
               algorithm, and they are:
                1. bk_vec_data
                2. dk_vec_data
                3. W_hat
                4. A boolean
results2     = the optimization object of 2-step GMM estimation
---------------------------------------------------------------------------
'''
# get 2-step params input
params_2step = np.append(lambda_gmm, delta_gmm[:-1])
# get 2-step W matrix
# get rid of the first element of dk becasue dk_0 = 1 - bk_0
data_moms = np.append(bk_vec_data, dk_vec_data[1:])
gmm_moms = np.append(bk_vec_gmm, dk_vec_gmm[1:])
err2 = (data_moms - gmm_moms).reshape(47,1)
VCV2 = np.dot(err2, err2.T)*(data.shape[0]**-1)
# Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
W_hat2 = np.linalg.pinv(VCV2)# get 2-step arguments
gmm_args2 = (bk_vec_data, dk_vec_data, W_hat2, months, True)

# run optimization
results2 = opt.minimize(criterion, params_2step, args=(gmm_args), \
                        method='SLSQP', constraints=cons, \
                        bounds=bounds)

'''
---------------------------------------------------------------------------
1.4. Report GMM outcomes, 2-step
---------------------------------------------------------------------------
lambda_gmm2  = a numpy array of (5,) lambda0 to lambda4
lambda0_gmm2 = the gmm estimator of lambda0
lambda1_gmm2 = the gmm estimator of lambda1
lambda2_gmm2 = the gmm estimator of lambda2
lambda3_gmm2 = the gmm estimator of lambda3
lambda4_gmm2 = the gmm estimator of lambda4
delta1_gmm2  = the gmm estimator of delta1
delta2_gmm2  = the gmm estimator of delta2
delta3_gmm2  = the gmm estimator of delta3
delta4_gmm2  = the gmm estimator of delta4
delta_gmm2   = a numpy array of (4.) delta1 to delta4
bk_vec_gmm2  = the model moments bk using the gmm estimators
dk_vec_gmm2  = the model moments dk using the gmm estimators
---------------------------------------------------------------------------
'''
# unpack and organize results
lambda_gmm2 = results2.x[:5]
lambda0_gmm2, lambda1_gmm2, lambda2_gmm2, lambda3_gmm2, lambda4_gmm2 = \
                                                                    lambda_gmm2
delta1_gmm2, delta2_gmm2, delta3_gmm2 = results2.x[5:]
delta4_gmm2 = 1 - lambda0_gmm2 - delta1_gmm2 - delta2_gmm2 - delta3_gmm2
delta_gmm2 = np.array([delta1_gmm2, delta2_gmm2, delta3_gmm2, delta4_gmm2])
# get model moments
bk_vec_gmm2, dk_vec_gmm2 = model_moments(lambda_gmm2, delta_gmm2[:-1], months)

# report GMM parameters
print('----------------------------------------------------------------------')
print('2-Step GMM Estimation')
print('----------------------------------------------------------------------')
print('Optimization Result')
print(results2)
print('----------------------------------------------------------------------')
print('GMM Estimators: Lambdas')
print(lambda_gmm2)
print('GMM Estimators: Deltas')
print(delta_gmm2)
print('----------------------------------------------------------------------')
print('Data Moments vs. Model Moments')
print('----------------------------------------------------------------------')
print('bk data')
print(bk_vec_data)
print('bk_gmm')
print(bk_vec_gmm2)
print('-------------------------------------')
print('dk_data')
print(dk_vec_data)
print('dk_gmm')
print(dk_vec_gmm2)
print('----------------------------------------------------------------------')
print()
print()

'''
---------------------------------------------------------------------------
Part 2. Esimating individual health care expenditures:
    2.1. Labeling positive expensitures
    2.2. Explore positive expenditure types
    2.3. Data preparation for cost estimation (note complete)
    2.4. Cost estimation (not complete)
---------------------------------------------------------------------------
2.1. Labeling positive expensitures
---------------------------------------------------------------------------
'''
# get spell probabilities
censored_none2, censored_none3, censored_none4 = \
    spell_prob(lambda_gmm2, delta_gmm2, data.shape[1], 'none')
censored_front2, censored_front3, censored_front4 = \
    spell_prob(lambda_gmm2, delta_gmm2, data.shape[1], 'front')
censored_back2, censored_back3, censored_back4 = \
    spell_prob(lambda_gmm2, delta_gmm2, data.shape[1], 'back')
censored_both2, censored_both3, censored_both4 = \
    spell_prob(lambda_gmm2, delta_gmm2, data.shape[1], 'both')


# get predicted labels for different spell lengths and censore types
label_uncensored = [] # uncensored
for i in range(22):
    if censored_none2[i] > censored_none3[i] and censored_none2[i] > censored_none4[i]:
        label_uncensored.append(2)
    elif censored_none3[i] > censored_none2[i] and censored_none3[i] > censored_none4[i]:
        label_uncensored.append(3)
    elif censored_none4[i] > censored_none2[i] and censored_none4[i] > censored_none3[i]:
        label_uncensored.append(4)
label_fcensored = [] # front-censored
for i in range(23):
    if censored_front2[i] > censored_front3[i] and censored_front2[i] > censored_front4[i]:
        label_fcensored.append(2)
    elif censored_front3[i] > censored_front2[i] and censored_front3[i] > censored_front4[i]:
        label_fcensored.append(3)
    elif censored_front4[i] > censored_front2[i] and censored_front4[i] > censored_front3[i]:
        label_fcensored.append(4)
label_bcensored = [] # back-censored
for i in range(23):
    if censored_back2[i] > censored_back3[i] and censored_back2[i] > censored_back4[i]:
        label_bcensored.append(2)
    elif censored_back3[i] > censored_back2[i] and censored_back3[i] > censored_back4[i]:
        label_bcensored.append(3)
    elif censored_back4[i] > censored_back2[i] and censored_back4[i] > censored_back3[i]:
        label_bcensored.append(4)
label_2censored = [] # front- and back-censored
for i in range(24):
    if censored_both2[i] > censored_both3[i] and censored_both2[i] > censored_both4[i]:
        label_2censored.append(2)
    elif censored_both3[i] > censored_both2[i] and censored_both3[i] > censored_both4[i]:
        label_2censored.append(3)
    elif censored_both4[i] > censored_both2[i] and censored_both4[i] > censored_both3[i]:
        label_2censored.append(4)


# get spell length index for different censor types
spell = data_nonneg.iloc[:,3:].copy(deep=True)
spell = np.where(spell > 0, 1, 0)

spell_fcensored = np.zeros_like(spell) # front-censored
spell_bcensored = np.zeros_like(spell) # back-censored
for i in range(spell.shape[1]-1):
    fcrit = np.array(spell[:,i] != 0) * np.array(spell.sum(axis=1) != 24)
    bcrit = np.array(spell[:,-(1+i)] != 0) * np.array(spell.sum(axis=1) != 24)
    for j in range(i+1):
        fcritj = np.array(spell[:,i-j] != 0)
        fcrit = fcrit * fcritj
        bcritj = np.array(spell[:,-(1+i-j)] != 0)
        bcrit = bcrit * bcritj
    spell_fcensored[fcrit,i] = 1
    spell_bcensored[bcrit,-(1+i)] = 1

spell_2censored = np.zeros_like(spell) # front- and back-censored
for i in range(spell.shape[1]):
    crit = np.array(spell.sum(axis=1) == spell.shape[1])
    spell_2censored[crit,i] = 1

spell_uncensored = np.zeros_like(spell) # uncensored
for i in range(1,spell.shape[1]-1):
    crit = np.array(spell[:,i] != 0)
    crit = crit * np.array(spell_fcensored[:,i] == 0)
    crit = crit * np.array(spell_bcensored[:,i] == 0)
    crit = crit * np.array(spell_2censored[:,i] == 0)
    spell_uncensored[crit,i] = 1

# spell length index
spell_fcensored2 = np.zeros_like(spell) # front-censored
spell_bcensored2 = np.zeros_like(spell) # back-censored
for i in range(spell.shape[1]):
    fcrit = np.array(spell_fcensored[:,i] != 0)
    bcrit = np.array(spell_bcensored[:,i] != 0)
    spell_fcensored2[fcrit,i] = spell_fcensored[fcrit,:].sum(axis=1)
    spell_bcensored2[bcrit,i] = spell_bcensored[bcrit,:].sum(axis=1)

spell_2censored2 = np.zeros_like(spell) # front- and back-censored
for i in range(spell.shape[1]):
    crit = np.array(spell.sum(axis=1) == spell.shape[1])
    spell_2censored2[crit,i] = 24

spell_uncensored2 = spell_uncensored.copy() # uncensored
for i in range(spell_uncensored2.shape[0]):
    for j in range(spell_uncensored2.shape[1]):
        if spell_uncensored[i,j] != 0:
            for k in range(j+1):
                if spell_uncensored2[i,j-k] == 0:
                    break
                else:
                    spell_uncensored2[i,j-k:j] = k+1
                    spell_uncensored2[i,j] += 1
            spell_uncensored2[i,j] -= 1


# connect predicted labels to different spell lenghts and censor types
label_spell_2censored = np.zeros_like(spell) # front- and back-censored
for i in range(spell.shape[1]):
    crit = np.array(spell.sum(axis=1) == spell.shape[1])
    label_spell_2censored[crit,i] = label_2censored[23]

label_spell_fcensored = np.zeros_like(spell) # front-censored
for i in range(spell.shape[0]):
    for j in range(spell.shape[1]):
        if spell_fcensored2[i,j] !=0:
            for k in range(len(label_fcensored)):
                if spell_fcensored2[i,j] == k+1:
                    label_spell_fcensored[i,j] = label_fcensored[k]

label_spell_bcensored = np.zeros_like(spell) # back-censored
for i in range(spell.shape[0]):
    for j in range(spell.shape[1]):
        if spell_bcensored2[i,j] !=0:
            for k in range(len(label_bcensored)):
                if spell_bcensored2[i,j] == k+1:
                    label_spell_bcensored[i,j] = label_bcensored[k]

label_spell_uncensored = np.zeros_like(spell) # uncensored
for i in range(spell.shape[0]):
    for j in range(spell.shape[1]):
        if spell_uncensored2[i,j] !=0:
            for k in range(len(label_uncensored)):
                if spell_uncensored2[i,j] == k+1:
                    label_spell_uncensored[i,j] = label_uncensored[k]


# get state index [h2, h3, h4]
h2_index = np.zeros_like(spell) # h2 index
for i in range(spell.shape[0]):
    for j in range(spell.shape[1]):
        if label_spell_2censored[i,j] == 2 or label_spell_fcensored[i,j] == 2 or \
        label_spell_bcensored[i,j] == 2 or label_spell_uncensored[i,j] == 2:
            h2_index[i,j] = 1

h3_index = np.zeros_like(spell) # h3 index
for i in range(spell.shape[0]):
    for j in range(spell.shape[1]):
        if label_spell_2censored[i,j] == 3 or label_spell_fcensored[i,j] == 3 or \
        label_spell_bcensored[i,j] == 3 or label_spell_uncensored[i,j] == 3:
            h3_index[i,j] = 1

h4_index = np.zeros_like(spell) # h4 index
for i in range(spell.shape[0]):
    for j in range(spell.shape[1]):
        if label_spell_2censored[i,j] == 4 or label_spell_fcensored[i,j] == 4 or \
        label_spell_bcensored[i,j] == 4 or label_spell_uncensored[i,j] == 4:
            h4_index[i,j] = 1

# remove rows where I made mistake :-(; I found typos and fixed them!
# crit = (spell.sum(axis=1) == (h2_index + h3_index + h4_index).sum(axis=1)).sum()
# h2_index2 = h2_index[crit,:]
# h3_index2 = h3_index[crit,:]
# h4_index2 = h4_index[crit,:]

'''
---------------------------------------------------------------------------
2.2. Explore positive expenditure types
---------------------------------------------------------------------------
'''
# get exp arrays for h2, h3, and h4
exp = data_nonneg.iloc[:,3:].copy(deep=True)
# exp = np.array(exp)[crit,:] # remove rows where I made mistake :-(; fixed
exp_h2 = exp * h2_index # an array of h2 expenditures only
exp_h3 = exp * h3_index # an array of h3 expenditures only
exp_h4 = exp * h4_index # an array of h4 expenditures only

# df for h2, h3, and h4
data_h2 = data_nonneg.copy(deep=True)
data_h2.iloc[:,3:] = data_h2.iloc[:,3:] * h2_index
data_h3 = data_nonneg.copy(deep=True)
data_h3.iloc[:,3:] = data_h3.iloc[:,3:] * h3_index
data_h4 = data_nonneg.copy(deep=True)
data_h4.iloc[:,3:] = data_h4.iloc[:,3:] * h4_index


# count, total
count_h2 = exp_h2[exp_h2!=0].count().sum()
count_h3 = exp_h3[exp_h3!=0].count().sum()
count_h4 = exp_h4[exp_h4!=0].count().sum()
# count, female
data_h2_female = data_h2[data_h2.male != True][data_h2.iloc[:,3:] != 0]
data_h3_female = data_h3[data_h3.male != True][data_h3.iloc[:,3:] != 0]
data_h4_female = data_h4[data_h4.male != True][data_h4.iloc[:,3:] != 0]
count_h2_female = data_h2_female.iloc[:,3:].count().sum()
count_h3_female = data_h3_female.iloc[:,3:].count().sum()
count_h4_female = data_h4_female.iloc[:,3:].count().sum()
# count, male
data_h2_male = data_h2[data_h2.male == True][data_h2.iloc[:,3:] != 0]
data_h3_male = data_h3[data_h3.male == True][data_h3.iloc[:,3:] != 0]
data_h4_male = data_h4[data_h4.male == True][data_h4.iloc[:,3:] != 0]
count_h2_male = data_h2_male.iloc[:,3:].count().sum()
count_h3_male = data_h3_male.iloc[:,3:].count().sum()
count_h4_male = data_h4_male.iloc[:,3:].count().sum()
# count, age<21
data_h2_age21 = data_h2[data_h2.age<21][data_h2.iloc[:,3:] != 0]
data_h3_age21 = data_h3[data_h3.age<21][data_h3.iloc[:,3:] != 0]
data_h4_age21 = data_h4[data_h4.age<21][data_h4.iloc[:,3:] != 0]
count_h2_age21 = data_h2_age21.iloc[:,3:].count().sum()
count_h3_age21 = data_h3_age21.iloc[:,3:].count().sum()
count_h4_age21 = data_h4_age21.iloc[:,3:].count().sum()
# count, age<45
data_h2_age21_45 = data_h2[data_h2.age>=21][data_h2.age<45][data_h2.iloc[:,3:] != 0]
data_h3_age21_45 = data_h3[data_h3.age>=21][data_h3.age<45][data_h3.iloc[:,3:] != 0]
data_h4_age21_45 = data_h4[data_h4.age>=21][data_h4.age<45][data_h4.iloc[:,3:] != 0]
count_h2_age21_45 = data_h2_age21_45.iloc[:,3:].count().sum()
count_h3_age21_45 = data_h3_age21_45.iloc[:,3:].count().sum()
count_h4_age21_45 = data_h4_age21_45.iloc[:,3:].count().sum()
# count, age<65
data_h2_age45_65 = data_h2[data_h2.age>=45][data_h2.age<65][data_h2.iloc[:,3:] != 0]
data_h3_age45_65 = data_h3[data_h3.age>=45][data_h3.age<65][data_h3.iloc[:,3:] != 0]
data_h4_age45_65 = data_h4[data_h4.age>=45][data_h4.age<65][data_h4.iloc[:,3:] != 0]
count_h2_age45_65 = data_h2_age45_65.iloc[:,3:].count().sum()
count_h3_age45_65 = data_h3_age45_65.iloc[:,3:].count().sum()
count_h4_age45_65 = data_h4_age45_65.iloc[:,3:].count().sum()
# count, age>65
data_h2_age65 = data_h2[data_h2.age>=65][data_h2.iloc[:,3:] != 0]
data_h3_age65 = data_h3[data_h3.age>=65][data_h3.iloc[:,3:] != 0]
data_h4_age65 = data_h4[data_h4.age>=65][data_h4.iloc[:,3:] != 0]
count_h2_age65 = data_h2_age65.iloc[:,3:].count().sum()
count_h3_age65 = data_h3_age65.iloc[:,3:].count().sum()
count_h4_age65 = data_h4_age65.iloc[:,3:].count().sum()

# mean, total
mean_h2 = exp_h2[exp_h2!=0].sum().sum() / count_h2
mean_h3 = exp_h3[exp_h3!=0].sum().sum() / count_h3
mean_h4 = exp_h4[exp_h4!=0].sum().sum() / count_h4
# mean, female
mean_h2_female = data_h2_female.iloc[:,3:].sum().sum() / count_h2_female
mean_h3_female = data_h3_female.iloc[:,3:].sum().sum() / count_h3_female
mean_h4_female = data_h4_female.iloc[:,3:].sum().sum() / count_h4_female
# mean, male
mean_h2_male = data_h2_male.iloc[:,3:].sum().sum() / count_h2_male
mean_h3_male = data_h3_male.iloc[:,3:].sum().sum() / count_h3_male
mean_h4_male = data_h4_male.iloc[:,3:].sum().sum() / count_h4_male
# mean, age<21
mean_h2_age21 = data_h2_age21.sum().sum() / count_h2_age21
mean_h3_age21 = data_h3_age21.sum().sum() / count_h3_age21
mean_h4_age21 = data_h4_age21.sum().sum() / count_h4_age21
# mean, age<45
mean_h2_age21_45 = data_h2_age21_45.sum().sum() / count_h2_age21_45
mean_h3_age21_45 = data_h3_age21_45.sum().sum() / count_h3_age21_45
mean_h4_age21_45 = data_h4_age21_45.sum().sum() / count_h4_age21_45
# mean, age<65
mean_h2_age45_65 = data_h2_age45_65.sum().sum() / count_h2_age45_65
mean_h3_age45_65 = data_h3_age45_65.sum().sum() / count_h3_age45_65
mean_h4_age45_65 = data_h4_age45_65.sum().sum() / count_h4_age45_65
# mean, age>65
mean_h2_age65 = data_h2_age65.sum().sum() / count_h2_age65
mean_h3_age65 = data_h3_age65.sum().sum() / count_h3_age65
mean_h4_age65 = data_h4_age65.sum().sum() / count_h4_age65

# standard deviation, total
std_h2 = np.sqrt(((exp_h2[exp_h2!=0]-mean_h2)**2).sum().sum()/count_h2)
std_h3 = np.sqrt(((exp_h3[exp_h3!=0]-mean_h3)**2).sum().sum()/count_h3)
std_h4 = np.sqrt(((exp_h4[exp_h4!=0]-mean_h4)**2).sum().sum()/count_h4)
# std, female
std_h2_female = np.sqrt(((data_h2_female.iloc[:,3:]-mean_h2_female)**2).sum().sum()/count_h2_female)
std_h3_female = np.sqrt(((data_h3_female.iloc[:,3:]-mean_h3_female)**2).sum().sum()/count_h3_female)
std_h4_female = np.sqrt(((data_h4_female.iloc[:,3:]-mean_h4_female)**2).sum().sum()/count_h4_female)
# std, male
std_h2_male = np.sqrt(((data_h2_male.iloc[:,3:]-mean_h2_male)**2).sum().sum()/count_h2_male)
std_h3_male = np.sqrt(((data_h3_male.iloc[:,3:]-mean_h3_male)**2).sum().sum()/count_h3_male)
std_h4_male = np.sqrt(((data_h4_male.iloc[:,3:]-mean_h4_male)**2).sum().sum()/count_h4_male)
# std, age<21
std_h2_age21 = np.sqrt(((data_h2_age21.iloc[:,3:]-mean_h2_age21)**2).sum().sum()/count_h2_age21)
std_h3_age21 = np.sqrt(((data_h3_age21.iloc[:,3:]-mean_h3_age21)**2).sum().sum()/count_h3_age21)
std_h4_age21 = np.sqrt(((data_h4_age21.iloc[:,3:]-mean_h4_age21)**2).sum().sum()/count_h4_age21)
# std, age<45
std_h2_age21_45 = np.sqrt(((data_h2_age21_45.iloc[:,3:]-mean_h2_age21_45)**2).sum().sum()/count_h2_age21_45)
std_h3_age21_45 = np.sqrt(((data_h3_age21_45.iloc[:,3:]-mean_h3_age21_45)**2).sum().sum()/count_h3_age21_45)
std_h4_age21_45 = np.sqrt(((data_h4_age21_45.iloc[:,3:]-mean_h4_age21_45)**2).sum().sum()/count_h4_age21_45)
# std, age<65
std_h2_age45_65 = np.sqrt(((data_h2_age45_65.iloc[:,3:]-mean_h2_age45_65)**2).sum().sum()/count_h2_age45_65)
std_h3_age45_65 = np.sqrt(((data_h3_age45_65.iloc[:,3:]-mean_h3_age45_65)**2).sum().sum()/count_h3_age45_65)
std_h4_age45_65 = np.sqrt(((data_h4_age45_65.iloc[:,3:]-mean_h4_age45_65)**2).sum().sum()/count_h4_age45_65)
# std, age>65
std_h2_age65 = np.sqrt(((data_h2_age65.iloc[:,3:]-mean_h2_age65)**2).sum().sum()/count_h2_age65)
std_h3_age65 = np.sqrt(((data_h3_age65.iloc[:,3:]-mean_h3_age65)**2).sum().sum()/count_h3_age65)
std_h4_age65 = np.sqrt(((data_h4_age65.iloc[:,3:]-mean_h4_age65)**2).sum().sum()/count_h4_age65)

# report
print('----------------------------------------------------------------------')
print('Descriptive stats of positive health expenditure types')
print('----------------------------------------------------------------------')
print('Counts--All')
print('H2:', count_h2, 'H3:', count_h3, 'H4:', count_h4)
print('Counts--Female' )
print('H2:', count_h2_female, 'H3:', count_h3_female, 'H4:', count_h4_female)
print('Counts--Male' )
print('H2:', count_h2_male, 'H3:', count_h3_male, 'H4:', count_h4_male)
print('Counts--Age < 21' )
print('H2:', count_h2_age21, 'H3:', count_h3_age21, 'H4:', count_h4_age21)
print('Counts--21 <= Age < 45' )
print('H2:', count_h2_age21_45, 'H3:', count_h3_age21_45, 'H4:', count_h4_age21_45)
print('Counts--45 <= Age < 65' )
print('H2:', count_h2_age45_65, 'H3:', count_h3_age45_65, 'H4:', count_h4_age45_65)
print('Counts--Age >= 65' )
print('H2:', count_h2_age65, 'H3:', count_h3_age65, 'H4:', count_h4_age65)
print('----------------------------------------------------------------------')
print('Mean--All')
print('H2:', mean_h2, 'H3:', mean_h3, 'H4:', mean_h4)
print('Mean--Female' )
print('H2:', mean_h2_female, 'H3:', mean_h3_female, 'H4:', mean_h4_female)
print('Mean--Male' )
print('H2:', mean_h2_male, 'H3:', mean_h3_male, 'H4:', mean_h4_male)
print('Mean--Age < 21' )
print('H2:', mean_h2_age21, 'H3:', mean_h3_age21, 'H4:', mean_h4_age21)
print('Mean--21 <= Age < 45' )
print('H2:', mean_h2_age21_45, 'H3:', mean_h3_age21_45, 'H4:', mean_h4_age21_45)
print('Mean--45 <= Age < 65' )
print('H2:', mean_h2_age45_65, 'H3:', mean_h3_age45_65, 'H4:', mean_h4_age45_65)
print('Mean--Age >= 65' )
print('H2:', mean_h2_age65, 'H3:', mean_h3_age65, 'H4:', mean_h4_age65)
print('----------------------------------------------------------------------')
print('Standard deviation--All')
print('H2:', std_h2, 'H3:', std_h3, 'H4:', std_h4)
print('Standard deviation--Female' )
print('H2:', std_h2_female, 'H3:', std_h3_female, 'H4:', std_h4_female)
print('Standard deviation--Male' )
print('H2:', std_h2_male, 'H3:', std_h3_male, 'H4:', std_h4_male)
print('Standard deviation--Age < 21' )
print('H2:', std_h2_age21, 'H3:', std_h3_age21, 'H4:', std_h4_age21)
print('Standard deviation--21 <= Age < 45' )
print('H2:', std_h2_age21_45, 'H3:', std_h3_age21_45, 'H4:', std_h4_age21_45)
print('Standard deviation--45 <= Age < 65' )
print('H2:', std_h2_age45_65, 'H3:', std_h3_age45_65, 'H4:', std_h4_age45_65)
print('Standard deviation--Age >= 65' )
print('H2:', std_h2_age65, 'H3:', std_h3_age65, 'H4:', std_h4_age65)
print('----------------------------------------------------------------------')
print()
print()

'''
---------------------------------------------------------------------------
2.3. Preparation for cost estimation; could not complete
---------------------------------------------------------------------------
'''
# # get err_vec and criterion functions
# def err_vec(data, rho_vec, mu_vec, gamma_vec, h_vec):
#     rho2, rho3, rho4 = rho_vec
#     mu2, mu3, mu4 = mu_vec
#     gamma2, gamma3, gamma4 = gamma_vec
#     err_vec = []
#     for h in h_vec:
#         if h == 'h2':
#             exp = data * h2_index2
#             # mu = exp.sum()/h2_index2.sum()
#             mu = mu2
#             gamma = gamma2
#             rho = rho2
#         elif h == 'h3':
#             exp = data * h3_index2
#             # mu = exp.sum()/h3_index2.sum()
#             mu = mu3
#             gamma = gamma3
#             rho = rho3
#         elif h == 'h4':
#             exp = data * h4_index2
#             # mu = exp.sum()/h4_index2.sum()
#             mu = mu4
#             gamma = gamma4
#             rho = rho4
#
#         err = np.zeros_like(exp)
#         for i in range(exp.shape[0]):
#             for t in range(exp.shape[1]):
#                 if (t == 0 or exp[i,t-1] == 0) and exp[i,t] != 0:
#                     err[i,t] = mu + gamma * (t-1)
#                 elif exp[i,t-1] > 0 and exp[i,t] != 0:
#                     err[i,t] = np.log(exp[i,t]) - rho * np.log(exp[i,t-1]) - (1+rho)*(mu + gamma*(t-1))
#         err_vec.append(err.flatten().sum())
#
#     return np.array(err_vec)
#
# def est_criterion(params, *args):
#     rho2, rho3, rho4, mu2, mu3, mu4, gamma2, gamma3, gamma4 = params
#     data, h_vec, W = args
#
#     rho_vec = rho2, rho3, rho4
#     mu_vec = mu2, mu3, mu4
#     gamma_vec = gamma2, gamma3, gamma4
#
#     err = err_vec(data, rho_vec, mu_vec, gamma_vec, h_vec)
#
#     crit_val = np.dot(np.dot(err.T, W), err)
#
#     return crit_val

'''
---------------------------------------------------------------------------
2.4. Cost estimation; could not complete
---------------------------------------------------------------------------
'''
# rho2_init, rho3_init, rho4_init = np.array([.1, .1, .1])
# mu2_init = exp_h2.sum()/h2_index.sum()
# mu3_init = exp_h3.sum()/h3_index.sum()
# mu4_init = exp_h4.sum()/h4_index.sum()
# gamma2_init, gamma3_init, gamma4_init = np.array([.2, .2, .2])
# params_est = np.array([rho2_init, rho3_init, rho4_init,
#     mu2_init, mu3_init, mu4_init,
#     gamma2_init, gamma3_init, gamma4_init])
#
# h_est = ['h2', 'h3', 'h4']
# W_hat = np.eye(3)
# est_args = (exp, h_est, W_hat)
#
# bounds = ((-1, 1),(-1, 1),(-1, 1),
#         (0,None),(0,None),(0,None),
#         (None,None),(None,None),(None,None))
#
# # doesn't work...
# results3 = \
#     opt.minimize(est_criterion, params_est, args=(est_args),
#                  method='SLSQP',
#                 #  constraints=cons,
#                  bounds=bounds,
#                  options={'disp': True})
