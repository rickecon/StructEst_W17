# MACS 40200: PS3
# Name: Dongping Zhang
# Python Version: 3.5
# Seed: None

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
Exercise 1/a.: Use the data (w_t, k_t) and equation (3) and (5) to estimate
               the four parameters (alpha, rho, mu, sigma) by MLE.

               Here is a brief model description of variables:
               c_t   = aggregate consumption in period t
               k_t+1 = total household savings and investment in period t for 
                       which they receive a return in the next period
               w_t   = wage per unit of labor in period t
               r_t   = interest rate or rate of return on investment 
               z_t   = total factor productivity
------------------------------------------------------------------------------
macro        = MacroSeries.txt dataset with added column names: cc, kk, ww, rr
get_zz       = a function to get z based on variables w and k, and parameter 
               alpha
log_lik_norm = a function to compute log-likelihood values
crit_norm    = a function that is the objective function
mle_args     = variables that would be used during MLE optimizations
params_init  = an numpy array that are the initial guesses of the four 
               parameters
results      = the optimization results object
alpha_MLE1   = the MLE estimator of alpha
rho_MLE1     = the MLE estimator of rho
mu_MLE1      = the MLE estimator of mu
sigma_MLE1   = the MLE estimator of sigma
vcov1        = the variance-covariance matrix of the optimization
------------------------------------------------------------------------------
'''
# load the MacroSeries.txt raw data through pandas
macro = pd.read_table('MacroSeries.txt', sep = ',', header = 0, \
                      names = ['cc', 'kk', 'ww', 'rr'])

# Assumptions:
# 1. first observation in the data file is t = 1
# 2. k_1 = first observation in the data file for the variable k_t
# 3. z_0 = mu so that z_1 = mu
# 4. discount factor beta = 0.99

# linear transformation of equation(3) to get z_t
def get_zz(ww, kk, alpha_param):
    '''
    --------------------------------------------------------------------
    Compute variable z based on variables w and k and parameter alpha
    --------------------------------------------------------------------
    INPUTS:
    ww          = wage per unit of labor in period t
    kk          = total household savings and investment in period t for 
                  which they receive a return in the next period
    alpha_param = the alpha parameter
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    zz          = total factor productivity
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: zz
    --------------------------------------------------------------------
    '''
    zz = np.log( ww / ( (1 - alpha_param) * (kk)**alpha_param) )
    return zz


def log_lik_norm(zz, rho, mu, sigma):
    '''
    --------------------------------------------------------------------
    Compute the log-likelihood value for z_t given normal distribution
    parameters rho, mu, and sigma.
    --------------------------------------------------------------------
    INPUTS:
    zz    = total factor productivity at time period t
    rho   = a parameter of z
    mu    = a parameter of z
    sigma = the standard deviation of the error term

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    zz_lag      = z_t-1 which is the z value of the pervious period
    pdf_vals    = (N,) vector, the probability of observing variable z
                  at time t based on parameters rho, mu, and sigma 
    ln_pdf_vals = (N,) vector, natural logarithm of normal PDF values
                  of z at time t based on parameters rho, mu, and sigma 
    log_lik_val = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    # deal with z_t-1
    zz_lag = zz.shift()
    zz_lag[0] = mu

    # compute pdf vals
    # z_t ~ N(rho * z_t-1 + (1 - rho)mu, sigma^2)
    pdf_vals = ( 1 / (sigma * np.sqrt(2 * np.pi) ) * \
                np.exp( - ( zz - (rho * zz_lag + (1 - rho) * mu) )**2 /\
                        (2 * sigma**2)) )

    # compute the log_likelihood values 
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val 


def crit_norm(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (4,) vector, ([alpha, rho, mu, sigma])
    args   = length 2 tuple, (variable w, variable k)
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    get_zz()
    log_lik_norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    zz          = (N,) vector, generated by using variable w and k, and 
                  parameter alpha
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of log_lik_val
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    alpha, rho, mu, sigma = params
    ww, kk = args
    zz = get_zz(ww, kk, alpha)
    log_lik_val = log_lik_norm(zz, rho, mu, sigma)
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val


# create bounds for each parameter
mle_args = (macro.ww, macro.kk)
bounds = ((1e-10, 1-1e-10), (-1+1e-10, 1-1e-10), (1e-10, None), (1e-10, None))
params_init = np.array([0.5, 0, 0.2, 0.3])
results = opt.minimize(crit_norm, params_init, args=(mle_args), \
                       bounds = bounds, method = 'L-BFGS-B') 
alpha_MLE1, rho_MLE1, mu_MLE1, sigma_MLE1 = results.x
vcov1 = results.hess_inv.todense()


# print out MLE estimators
print('Problem 1/(a).')
print('alpha_MLE1: ', alpha_MLE1)
print('rho_MLE1: ', rho_MLE1)
print('mu_MLE1: ', mu_MLE1)
print('sigma_MLE1', sigma_MLE1)
print()
print(results)
print()
print('inverse hessian vcov matrix is:')
print(vcov1)
print()


'''
------------------------------------------------------------------------------
Exercise 1/b.: Use the data (r_t, k_t) and equation (4) and (5) to estimate
               the four parameters (alpha, rho, mu, sigma) by MLE.

               Here is a brief model description of variables:
               c_t   = aggregate consumption in period t
               k_t+1 = total household savings and investment in period t for 
                       which they receive a return in the next period
               w_t   = wage per unit of labor in period t
               r_t   = interest rate or rate of return on investment 
               z_t   = total factor productivity
------------------------------------------------------------------------------
get_zz2       = a function to get z based on variables r and k, and parameter 
                alpha
crit_norm2    = a function that is the objective function
mle_args2     = variables that would be used during MLE optimizations
params_init2  = an numpy array that are the initial guesses of the four 
                parameters
results2      = the optimization results object
alpha_MLE2    = the MLE estimator of alpha
rho_MLE2      = the MLE estimator of rho
mu_MLE2       = the MLE estimator of mu
sigma_MLE2    = the MLE estimator of sigma
vcov2         = the variance-covariance matrix of the optimization
------------------------------------------------------------------------------
'''
def get_zz2(rr, kk, alpha_param):
    '''
    --------------------------------------------------------------------
    Compute variable z based on variables r and k and parameter alpha
    --------------------------------------------------------------------
    INPUTS:
    rr          = interest rate or rate of return on investment at time t
    kk          = total household savings and investment in period t for which 
                  they receive a return in the next period
    alpha_param = the alpha parameter
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    zz          = total factor productivity
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: zz
    --------------------------------------------------------------------
    '''
    zz = np.log( rr / ( alpha_param * (kk)**(alpha_param - 1) ) )
    return zz


def crit_norm2(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (4,) vector, ([alpha, rho, mu, sigma])
    args   = length 2 tuple, (variable r, variable k)
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    get_zz2()
    log_lik_norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    zz          = (N,) vector, generated by using variable w and k, and 
                  parameter alpha
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of log_lik_val
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    alpha, rho, mu, sigma = params
    rr, kk = args
    zz = get_zz2(rr, kk, alpha)
    log_lik_val = log_lik_norm(zz, rho, mu, sigma)
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val


# create bounds for each parameter
mle_args2 = (macro.rr, macro.kk)
params_init2 = np.array([0.5, 0.5, 10, 1])
results2 = opt.minimize(crit_norm2, params_init2, args=(mle_args2), \
                        bounds = bounds, method = 'L-BFGS-B') 
alpha_MLE2, rho_MLE2, mu_MLE2, sigma_MLE2 = results2.x
vcov2 = results2.hess_inv.todense()


# print out MLE estimators
print('Problem 1/(b).')
print('alpha_MLE2: ', alpha_MLE2)
print('rho_MLE2: ', rho_MLE2)
print('mu_MLE2: ', mu_MLE2)
print('sigma_MLE2', sigma_MLE2)
print()
print(results2)
print()
print('inverse hessian vcov matrix is:')
print(vcov2)
print()


'''
------------------------------------------------------------------------------
Exercise 1/a.: Use the estimation from part (a), if k_t = 7500000m and
               z_t-1 = 10, what is the probability that the interest rate
               this period will be greater than r_t = 1
 
               Here is a brief model description of variables:
               c_t   = aggregate consumption in period t
               k_t+1 = total household savings and investment in period t for 
                       which they receive a return in the next period
               w_t   = wage per unit of labor in period t
               r_t   = interest rate or rate of return on investment 
               z_t   = total factor productivity
------------------------------------------------------------------------------
rr            = r_t, given value = 1, which is the interest rate of this period
kk            = k_t, given value = 7,500,000, which is the investment/savings
                of the current period
z_star        = the z value when r_t = 1
z_t_lag       = r_t-1, given value = 10, which is the productivity shock in
                the previous period
z_mu          = variable z follows a normal distribution, and this variable is
                the mean of the normal distribution and is constructed by MLE
                estimators from part (a)
z_sig         = variable z follows a normal distribution, and this variable is
                the sd of the normal distribution and is constructed by MLE
                estimators from part (a)
prob_z_gzStar = the probability to observe the interest rate this period to be
                greater than r_t = 1
------------------------------------------------------------------------------
'''
# compute for z_star using rr and kk
rr_ = 1
kk_ = 7500000
z_star = get_zz2(rr_, kk_, alpha_MLE1)

# solve the probability that z_t > z_star
# figure out z_t distribution
z_t_lag = 10
z_mu = rho_MLE1 * z_t_lag + (1 - rho_MLE1) * mu_MLE1
z_sig = sigma_MLE1
prob_z_gzStar = 1 - sts.norm.cdf(z_star, z_mu, z_sig)

# print out results
print('Problem 1/(c).')
print('z_star is: ', z_star)
print('Probability that z_t > z_star: ', prob_z_gzStar)
