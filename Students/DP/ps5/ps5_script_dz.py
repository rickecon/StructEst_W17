# MACS 40200: PS5
# Name: Dongping Zhang
# Python Version: 3.5
# Seed: 1234

import pandas as pd
import numpy as np
import scipy.stats as sts
import os
import matplotlib.pyplot as plt
import scipy.special as spc
import scipy.stats as sts
import scipy.optimize as opt
import scipy.integrate as integration
import math

'''
------------------------------------------------------------------------------
Exercise 1/a.: Estimate (alpha, beta, rho, mu) by GMM
 
               Here is a brief model description of variables:
               c_t   = aggregate consumption in period t
               k_t+1 = total household savings and investment in period t for 
                       which they receive a return in the next period
               w_t   = wage per unit of labor in period t
               r_t   = interest rate or rate of return on investment 
               z_t   = total factor productivity
------------------------------------------------------------------------------
macro         = MacroSeries.txt dataset with added column names: cc, kk, ww, rr
model_moments = a function that computes the model moments
err_vec       = a function that constructs the moment error vector
criterion     = the criterion function for the optimization procedure
params_init   = the initial guesses of all parameters
W_hat_I4      = an （4, 4）identity matrix 
GMM_args      = the input arguments for the optimization procedure
param_bounds  = the bounds of all parameters
results_GMM   = the returned optimization object
alpha_GMM     = GMM estimates of alpha
beta_GMM      = GMM estimates of beta
rho_GMM       = GMM estimates of rho
mu_GMM        = GMM estimates of mu
val_crit_GMM  = the value returned by the criterion when using GMM estimators
------------------------------------------------------------------------------
'''
# load the MacroSeries.txt raw data through pandas
macro = pd.read_table('MacroSeries.txt', sep = ',', header = 0, \
                      names = ['cc', 'kk', 'ww', 'rr'])

# Assumptions:
# 1. first observation in the data file is t = 1
# 2. k_1 = first observation in the data file for the variable k_t


def model_moments(rr, kk, cc, ww, *params):
    '''
    --------------------------------------------------------------------
    This function computes the four data moments for GMM
    --------------------------------------------------------------------
    INPUTS:
    rr: variable from the dataset
    kk: variable from the dataset 
    cc: variable from the dataset
    ww: variable from the dataset
    params: all parameters that include -- alpha, beta, rho, mu
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    zz   = series z
    mm1s = a list of simulated model moment 1
    mm2s = a list of simulated model momemt 2
    mm3s = a list of simulated model moment 3
    mm4s = a list of simulated model moment 4
    mm1  = real model moment 1
    mm2  = real model moment 2
    mm3  = real model moment 3
    mm4  = real model moemnt 4
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mm1, mm2, mm3, mm4
    --------------------------------------------------------------------
    '''
    # define all parameters
    alpha, beta, rho, mu = params

    # first need to back out a series for zz using equation 4
    zz = np.log( rr / ( alpha * (kk)**(alpha - 1) ) )

    # initialize model moments by setting them equal to 0
    mm1s = [] # eq6
    mm2s = [] # eq7
    mm3s = [] # eq8
    mm4s = [] # eq9
    for i in range(macro.shape[0] - 1):
        mm1s.append( zz[i + 1] - rho * zz[i] - (1 - rho) * mu ) 
        mm2s.append( ( zz[i + 1] - rho * zz[i] - (1 - rho) * mu ) * zz[i] )
        mm3s.append( beta * alpha * np.exp(zz[i + 1]) \
                     * kk[i + 1]**(alpha - 1) * cc[i] / cc[i + 1] - 1 )
        mm4s.append( ( beta * alpha * np.exp(zz[i + 1]) * 
                     kk[i + 1]**(alpha - 1) * cc[i] / cc[i + 1] - 1 ) * ww[i] )

    # transform those 4 moments to expected values
    mm1 = sum(mm1s) / len(mm1s) 
    mm2 = sum(mm2s) / len(mm2s)
    mm3 = sum(mm3s) / len(mm3s)
    mm4 = sum(mm4s) / len(mm4s)

    return mm1, mm2, mm3, mm4


def err_vec(rr, kk, cc, ww, *params): 
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    rr: variable from the dataset
    kk: variable from the dataset 
    cc: variable from the dataset
    ww: variable from the dataset
    params: all parameters that include -- alpha, beta, rho, mu
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments()
        model_moments()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mm1        = real model moment 1
    mm2        = real model moment 2
    mm3        = real model moment 3
    mm4        = real model moemnt 4
    moms_data  = a numpy array of data moments
    moms_model = a numpy array of model moments
    err_vec    = the error vector 
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    # define all parameters
    alpha, beta, rho, mu = params

    # compute data moments
    mm1, mm2, mm3, mm4 = model_moments(rr, kk, cc, ww, \
                                       alpha, beta, rho, mu)
    moms_data = np.array([ [mm1], [mm2], [mm3], [mm4] ])

    # indicate model moments
    moms_model = np.array([ [0], [0], [0], [0] ])

    # compute the error vector
    err_vec = moms_model - moms_data

    return err_vec


def criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params: all parameters that include -- alpha, beta, rho, mu 
    args: all required arguments -- rr, kk, cc, ww, W
 
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
 
    OBJECTS CREATED WITHIN FUNCTION:
    err        = column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    alpha, beta, rho, mu = params
    rr, kk, cc, ww, W = args

    err = err_vec(rr, kk, cc, ww, alpha, beta, rho, mu)
    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val
 

# initial guess of parameters
params_init = np.array( [0.5 ,0.5, 0, 1] )
# define weighting matrix as the identity matrix
W_hat_I4 = np.eye(4)
# all required components
GMM_args = (macro.rr, macro.kk, macro.cc, macro.ww, W_hat_I4)
# define params bounds
param_bounds = ( (1e-10, 1-1e-10), (1e-10, 1-1e-10) , \
                 (-1+1e-10, 1-1e-10), (1e-10, None) )

# implement optimization
results_GMM = opt.minimize(criterion, params_init, args=(GMM_args), \
                           bounds = param_bounds, method = 'L-BFGS-B', \
                           options={'eps': 1e-10}) 

# print out the results
alpha_GMM, beta_GMM, rho_GMM, mu_GMM = results_GMM.x

# val of minimized criterion func
val_crit_GMM = criterion([alpha_GMM, beta_GMM, rho_GMM, mu_GMM], \
                         macro.rr, macro.kk, macro.cc, macro.ww, W_hat_I4)

# print out GMM estimators
print('----------------------------------------------------------------------')
print('Problem 1/(a).')
print('alpha_GMM: ', alpha_GMM)
print('beta_GMM: ', beta_GMM)
print('rho_GMM: ', rho_GMM)
print('mu_GMM: ', mu_GMM)
print('-----------------------------------')
print('val_crit_GMM: ', results_GMM.fun)
print('-----------------------------------')
print(results_GMM)
print('----------------------------------------------------------------------')
print()
print()

'''
------------------------------------------------------------------------------
Exercise 2/a.: Estimate (alpha, beta, rho, mu) by SMM
 
               Here is a brief model description of variables:
               c_t   = aggregate consumption in period t
               k_t+1 = total household savings and investment in period t for 
                       which they receive a return in the next period
               w_t   = wage per unit of labor in period t
               r_t   = interest rate or rate of return on investment 
               z_t   = total factor productivity
------------------------------------------------------------------------------
simulated_draws   = simulation function
model_moments_SMM = compute model moments for SMM
data_moments_SMM  = compute data moments for SMM
vec_err_SMM       = compute err vector for SMM
criterion_SMM     = the criterion function SMM
params_init_SMM   = initial guesses for SMM
W_hat_I6          = an (6, 6) identitiy weighting matrix
SMM_args          = all required arguments for SMM
SMM_bounds        = parameter bounds for SMM
results_SMM       = optimization return object
alpha_SMM         = the estimated alpha by SMM
beta_SMM          = the estimated beta by SMM 
rho_SMM           = the estimated rho by SMM
mu_SMM            = the estimated mu by SMM
sigma_SMM         = the estimated sigma by SMM
val_crit_SMM      = val of criterion function when using all SMM estimators
------------------------------------------------------------------------------
'''
def simulated_draws(unif_vals, xvals, *params):
    '''
    --------------------------------------------------------------------
    This function simulate all data needed
    --------------------------------------------------------------------
    INPUTS:
    params: all parameters that include -- alpha, beta, rho, mu 
    args: all required arguments -- rr, kk, cc, ww, W
 
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
 
    OBJECTS CREATED WITHIN FUNCTION:
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: kk_sims, cc_sims
    --------------------------------------------------------------------
    '''
    alpha, beta, rho, mu, sigma = params
    cc, kk, ww, rr = xvals
    
    # two assumptions for t=1
    z_1 = mu
    k_1 = kk.mean()

    # dimension of the simulation matrix
    nrow = 100
    ncol = 1000
    
    # draw 100 normally distributed values of epsilon
    normed_errors = sts.norm.ppf(unif_vals, 0, sigma)

    ###### simulate zz using eq5 #####
    zz_sims = np.zeros( (nrow, ncol) )
    # first row are all z_1 fixed
    zz_sims[0,:] = z_1
    for i in range(1, nrow):
        zz_sims[i,:]  = rho * zz_sims[i-1,:] + \
                        (1 - rho) * mu + normed_errors[i,:]

    ##### simulate kk using eq10 #####
    kk_sims = np.zeros( (nrow+1, ncol) )
    # first row are all k_1 fixed
    kk_sims[0,:] = k_1
    for i in range(1, nrow+1):
        kk_sims[i,:] = alpha * beta * np.exp(zz_sims[i-1,:]) * \
                       kk_sims[i-1,:]**alpha

    ##### simulate ww using eq3 #####
    ww_sims = np.zeros( (nrow, ncol) )
    for i in range(nrow):
        ww_sims[i,:] = (1 - alpha) * np.exp(zz_sims[i,:]) * kk_sims[i,:]**alpha

    ##### simulate rr using eq4 #####
    rr_sims = np.zeros( (nrow, ncol) )
    for i in range(nrow):
        rr_sims[i,:] = alpha * np.exp(zz_sims[i,:]) * (kk_sims[i,:])**(alpha-1)

    ##### simulate cc using eq2 #####
    cc_sims = np.zeros( (nrow, ncol) )
    for i in range(nrow):
        cc_sims[i,:] = rr_sims[i,:] * kk_sims[i,:] + ww_sims[i,:] \
                       - kk_sims[i+1,:]

    # to convert kk_sims back
    kk_sims = kk_sims[:-1,:]
    
    return kk_sims, cc_sims


def model_moments_SMM(kk_sims, cc_sims):
    '''
    --------------------------------------------------------------------
    This function computes the model moments for SMM
    --------------------------------------------------------------------
    INPUTS:
    kk_sims
    cc_sims
 
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
 
    OBJECTS CREATED WITHIN FUNCTION:
    mm1, mm2, mm3, mm4, mm5, mm6
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mm1, mm2, mm3, mm4, mm5, mm6
    --------------------------------------------------------------------
    '''
    # 1st moment: mean of cc
    mm1 = cc_sims.mean(axis = 0).mean()
    # 2nd moment: mean of kk
    mm2 = kk_sims.mean(axis = 0).mean()
    # 3rd moment: variance of cc
    mm3 = cc_sims.var(axis = 0).mean()
    # 4th moment: variance of kk
    mm4 = kk_sims.var(axis = 0).mean()
    # 5th moment: correlation between cc and kk
    mm5s = []
    for i in range(kk_sims.shape[1]):
        mm5s.append( np.corrcoef(cc_sims[:,i], kk_sims[:,i])[0][1] )
    mm5 = np.array(mm5s).mean()
    # 6th moment: correlation between k_t and k_t+1
    mm6s = []
    for i in range(kk_sims.shape[1]):
        mm6s.append( np.corrcoef(kk_sims[:-1,i], kk_sims[1:,i])[0][1] )
    mm6 = np.array(mm6s).mean()

    return mm1, mm2, mm3, mm4, mm5, mm6


def data_moments_SMM(kk, cc):
    '''
    --------------------------------------------------------------------
    This function computes the model moments for SMM
    --------------------------------------------------------------------
    INPUTS:
    kk: macro.kk
    cc: macro.cc
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
 
    OBJECTS CREATED WITHIN FUNCTION:
    dm1, dm2, dm3, dm4, dm5, kk_lag, kk_series, dm6
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: dm1, dm2, dm3, dm4, dm5, dm6, dm7
    --------------------------------------------------------------------
    '''
    # 1st moment: mean of cc
    dm1 = np.mean(cc)
    # 2nd moment: mean of kk
    dm2 = np.mean(kk)
    # 3rd moment: variance of cc
    dm3 = np.var(cc)
    # 4th moment: variance of kk
    dm4 = np.var(kk)
    # 5th moment correlation between cc and kk
    dm5 = np.corrcoef(cc, kk)[0][1]
    # 6th moment correlation between k_t and k_t+1
    kk_lag = kk.shift(-1)[:-1]
    kk_series = kk[0:98]
    dm6 = sts.pearsonr(kk_lag, kk_series)[0]
    
    return dm1, dm2, dm3, dm4, dm5, dm6
        

def err_vec_SMM(kk, cc, kk_sims, cc_sims):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    kk: variable from the dataset 
    cc: variable from the dataset
    kk_sims: simulated kks
    cc_sims: simulated ccs
 
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments_SMM()
        model_moments_SMM()
    
    OBJECTS CREATED WITHIN FUNCTION:
    dmm1 - dmm6, mmm1 - mmm6, moms_data, moms_model, err_vec
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    # calc data moments
    dmm1, dmm2, dmm3, dmm4, dmm5, dmm6 = data_moments_SMM(kk, cc)
    moms_data = np.array([ [dmm1], [dmm2], [dmm3], [dmm4], [dmm5], [dmm6] ])

    # calc model moments
    mmm1, mmm2, mmm3, mmm4, mmm5, mmm6 = model_moments_SMM(kk_sims, cc_sims)
    moms_model = np.array([ [mmm1], [mmm2], [mmm3], [mmm4], [mmm5], [mmm6] ])

    err_vec = (moms_model - moms_data) / moms_data
    
    return err_vec


def criterion_SMM(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params: all parameters that include -- alpha, beta, rho, mu, sigma 
    args: all required arguments -- rr, kk, cc, ww, W_hat
 
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
 
    OBJECTS CREATED WITHIN FUNCTION:
    err        = column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    alpha, beta, rho, mu, sigma = params
    cc, kk, ww, rr, W_hat = args
    kk_sims, cc_sims = simulated_draws(unif_vals, (cc, kk, ww, rr), \
                                       alpha, beta, rho, mu, sigma)
    err = err_vec_SMM(kk, cc, kk_sims, cc_sims)
    crit_val = np.dot(np.dot(err.T, W_hat), err)

    return crit_val


# define all inputs to the optimization function
params_init_SMM = np.array( [0.5, 0.5, 0, 0.5, 0.5] )
W_hat_I6 = np.eye(6)

# simulate uniform values
np.random.seed(1234)
unif_vals = sts.uniform.rvs(0, 1, size = (100, 1000))
SMM_args = (macro.cc, macro.kk, macro.ww, macro.rr, W_hat_I6)
SMM_bounds = ( (0.01, 0.99), (0.01, 0.99), (-0.99, 0.99), \
               (-0.5, 1), (0.001, 1) )

# optimization
results_SMM = opt.minimize(criterion_SMM, params_init_SMM, args = (SMM_args), \
                           method = 'L-BFGS-B', \
                           bounds = SMM_bounds, \
                           options = {'eps':1e-10})

# print out the results
alpha_SMM, beta_SMM, rho_SMM, mu_SMM, sigma_SMM = results_SMM.x

# val of minimized criterion func
val_crit_SMM = criterion_SMM([alpha_SMM, beta_SMM, rho_SMM, \
                              mu_SMM, alpha_SMM], \
                             macro.cc, macro.kk, \
                             macro.ww, macro.rr, W_hat_I6)

# optimal model moments
kk_sims, cc_sims = simulated_draws(unif_vals, \
                                   (macro.cc, macro.kk, macro.ww, macro.rr), \
                                   alpha_SMM, beta_SMM, rho_SMM, mu_SMM, \
                                   sigma_SMM)

# compute model moments and data moments
model_moments = np.array(model_moments_SMM(kk_sims, cc_sims))
data_moments = np.array(data_moments_SMM(macro.kk, macro.cc))

# print out SMM estimators
print('----------------------------------------------------------------------')
print('Problem 2/(a).')
print('alpha_SMM: ', alpha_SMM)
print('beta_SMM: ', beta_SMM)
print('rho_SMM: ', rho_SMM)
print('mu_SMM: ', mu_SMM)
print('sigma_SMM', sigma_SMM)
print('-----------------------------------')
print('val_crit_SMM: ', results_SMM.fun)
print('-----------------------------------')
print('data_moments')
print(data_moments)
print('-----------------------------------')
print('model_moments')
print(model_moments)
print('-----------------------------------')
print('difference between model moments and data moments:')
print(data_moments - model_moments)
print('-----------------------------------')
print(results_SMM)
print('----------------------------------------------------------------------')
