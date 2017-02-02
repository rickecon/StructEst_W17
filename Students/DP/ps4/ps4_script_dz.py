# MACS 40200: PS4
# Name: Dongping Zhang
# Python Version: 3.5
# Seed: None

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter as fstr
from matplotlib.ticker import MultipleLocator
import scipy.stats as sts
import scipy.special as spc
import scipy.stats as sts
import scipy.optimize as opt
import scipy.integrate as intgr
import numpy.linalg as lin

'''
------------------------------------------------------------------------------
Exercise 1/a.: Plot a density histogram using U.S. household income data
------------------------------------------------------------------------------
incomes      = the raw dataframe of U.S. household incomes data
incomes_plot = Boolean, =True if make a plot of U.S. household incomes 
               distribution
------------------------------------------------------------------------------
'''
# load the raw income data into pandas dataframe
incomes = pd.read_table('usincmoms.txt', header = None,\
                        names = ['percent', 'midpoint'])
# convert the incomes into thousands
incomes.midpoint /= 1000

incomes_plot = True
if incomes_plot: 
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
 
    # initialite a plot
    fig, axes = plt.subplots()

    # 1. construct correct weights
    weights = incomes.percent
    weights[40] /= 10 # 41st bar is 10 times fatter, thus dividing by 10
    weights[41] /= 20 # 42nd bar is 20 times fatter, thus dividing by 20    
    # 2. construct bins
    bins = np.array(incomes.midpoint) - 2.5 # get the bin intervals
    bins[40] = 200 # 41st bin starting value 200
    bins[41] = 250 # 42nd bin starting value 250
    bins = np.append(bins, 350) # -2.5 thus adding the end value of 42nd bin

    # create histogram
    n, bin_cuts, patches = plt.hist(incomes.midpoint, bins = bins, \
                                    weights=weights, color = 'maroon')

    # add grid 
    minorLocator = MultipleLocator(1)
    axes.xaxis.set_minor_locator(minorLocator)  
    plt.grid(b=True, which='major', axis = 'both', color='0.65', linestyle='-')

    # set xlimit
    plt.xlim([0, 350])
    plt.ylim([0, 0.09])
    
    # adding plot title and xlabel and ylabel
    plt.title('Distribution of household incomes in the U.S.')    
    plt.xlabel(r'Household Income (in $1,000)')
    plt.ylabel(r'Percent of observations in bin')
    
    # generate plot in the designated directory
    output_path = os.path.join(output_dir, 'Fig_1')
    plt.savefig(output_path)   
    plt.close()  


'''
------------------------------------------------------------------------------
Exercise 1/b.:Using GMM to fit a lognormal distribution to the income data
------------------------------------------------------------------------------
incomes_dataMoments  = a function that computes income data moments
incomes_modelMoments = a function that computes income model moments
incomes_err_vec      = a function that constructs moment error vectors
incomes_criterion    = the criterion function used for parameter optimization
mu_init_LN           = initial testing value for parmater mu
sig_init_LN          = initial testing value for parameter sigma
params_init_LN       = a tuple of length 2 consisting of mu and sigma initial
                       guesses
W_hat_LN             = weighting matrix -- data moments on the diagonal
gmm_args_LN          = arguments needed for optimization
results_LN           = optimization result object
mu_GMM_LN            = the GMM estimator of parameter mu
sig_GMM_LN           = the GMM estimator of parameter sigma
val_crit_LN          = the value of GMM criterion function at the estimated
                       parameter values
hist_midpoints       = the midpoints of bins of income histogram
dist_pts             = xvalues used to simulate datapoints for plotting
dist_vals_LN         = simulated probability using GMM estimators
GMM_LN               = Boolean, =True if make a plot of incomes distribution 
                       and the probability density function of lognormal using 
                       GMM parameters
------------------------------------------------------------------------------
'''
def incomes_dataMoments(df):
    '''
    --------------------------------------------------------------------
    This function returns the 42 data moments for GMM
    --------------------------------------------------------------------
    INPUTS:
    df = a pandas dataframe, incomes
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    data_moments = a list of length 42 that consists of data moments
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: data_moments
    --------------------------------------------------------------------
    '''
    data_moments = list(df.percent)
    return data_moments


def incomes_modelMoments(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the 42 model moments for GMM
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        sts.lognorm.cdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    LN_cdf        = a lambda function of lognormal cdf 
    model_moments = a list of length 42 that contains model moments
    bins          = a list that contains the intervals of each bins in
                    the incomes histogram
    prob          = a model moment

    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: model_moments
    --------------------------------------------------------------------
    '''
    LN_cdf = lambda x: sts.lognorm.cdf(x, sigma, 0, mu)
    model_moments = []
    # create a variable called bins
    bins = list(range(0,205,5)) + [250,350]

    for index in range(len(bins)-1):
        prob = LN_cdf(bins[index + 1]) - \
               LN_cdf(bins[index])
        model_moments.append(prob)

    return model_moments


def incomes_err_vec(df, mu, sigma, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    df = a pandas dataframe, incomes
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        incomes_dataMoments()
        incomes_modelMoments()
    
    OBJECTS CREATED WITHIN FUNCTION:
    data_moments  = a list of 42 data moments
    moms_data     = same as data_moments but in the form of np.array
    model_moments = a list of 42 model moments
    moms_model    = same as model_moments but in the form of np.array
    err_vec       = column vector of two moment error
                    functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    # compute data moments
    data_moments = incomes_dataMoments(df)
    moms_data = np.array(data_moments)

    # compute model moments
    model_moments = incomes_modelMoments(mu, sigma)
    moms_model = np.array(model_moments)

    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data

    return err_vec


def incomes_criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    args   = length 2 tuple, (df, W)
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
 
    OBJECTS CREATED WITHIN FUNCTION:
    err        = column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    df, W = args
    err = incomes_err_vec(df, mu, sigma, simple = False)
    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val


# initial guess of the lognormal parameters
mu_init_LN = 50
sig_init_LN = 20
params_init_LN = np.array( [mu_init_LN, sig_init_LN] )
# define weighting matrix to be an identity matrix as required
W_hat_LN = np.diag(incomes.percent, 0)
gmm_args_LN = (incomes, W_hat_LN)
# implement optimization algorithm
results_LN = opt.minimize(incomes_criterion, params_init_LN, \
                          args = (gmm_args_LN), \
                          method = 'L-BFGS-B', \
                          bounds = ((None, None), (1e-10, None)))

# variable assignment of MLE parameters
mu_GMM_LN, sig_GMM_LN = results_LN.x
# compute value of GMM criterion function using MOM estimators
val_crit_LN = incomes_criterion([mu_GMM_LN, sig_GMM_LN], \
                                incomes, W_hat_LN)

# print out the results
print('----------------------------------------------------------------------')
print('Problem 1/b')
print('mu_GMM_LN = ', mu_GMM_LN)
print('sigma_GMM_LN = ', sig_GMM_LN)
print('val of GMM criterion function using GMM estimators: ', val_crit_LN)
print('-----------------------------------')
print(results_LN)
print('----------------------------------------------------------------------')
print()

# simulate some points for distribution plotting
hist_midpoints = np.array(incomes.midpoint[:-2])
dist_pts = np.append(hist_midpoints, [225.5, 300])
dist_vals_LN = incomes_modelMoments(mu_GMM_LN, sig_GMM_LN)
dist_vals_LN[40] /= 10
dist_vals_LN[41] /= 20

GMM_LN = True
if GMM_LN: 
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
    # initialite a plot
    fig, axes = plt.subplots()

    # create histogram
    n, bin_cuts, patches = plt.hist(incomes.midpoint, bins = bins, \
                                    weights=weights, color = 'maroon')
    # plot the lognormal distribution
    plt.plot(dist_pts, dist_vals_LN, linewidth=2, color='green', \
             label=r'$LN (\hat{\mu}_{GMM}, \hat{\sigma}_{GMM})$')
    plt.legend(loc='upper right')

    # add grid 
    minorLocator = MultipleLocator(1)
    axes.xaxis.set_minor_locator(minorLocator)  
    plt.grid(b=True, which='major', axis = 'both', color='0.65', linestyle='-')

    # set xlimit
    plt.xlim([0, 350])
    plt.ylim([0, 0.09])
    
    # adding plot title and xlabel and ylabel
    plt.title('Distribution of household incomes in the U.S.')    
    plt.xlabel(r'Household Income (in $1,000)')
    plt.ylabel(r'Percent of observations in bin')
    
    # generate plot in the designated directory
    output_path = os.path.join(output_dir, 'Fig_2')
    plt.savefig(output_path)   
    plt.close()  


'''
------------------------------------------------------------------------------
Exercise 1/c.:Using GMM to fit a Gamma distribution to the income data
------------------------------------------------------------------------------
incomes_dataMoments_GA  = a function that computes income data moments
incomes_modelMoments_GA = a function that computes income model moments
incomes_err_vec_GA      = a function that constructs moment error vectors
incomes_criterion_GA    = the criterion function used for params optimization
alpha_init              = initial testing value for parmater alpha
beta_init               = initial testing value for parameter beta
params_init_GA          = a tuple of length 2 consisting alpha and beta initial
                          guesses
W_hat_GA                = weighting matrix -- data moments on the diagonal
gmm_args_GA             = arguments needed for optimization
results_GA              = optimization result object
alpha_GMM               = the GMM estimator of parameter alpha
beta_GMM                = the GMM estimator of parameter beta
val_crit_GA             = the value of GMM criterion function at the estimated
                          parameter values
dist_vals_GA            = simulated probability using GMM estimators
GMM_GA                  = Boolean, =True if make a plot of incomes distribution 
                          and the probability density function of Gamma 
                          using GMM parameters
------------------------------------------------------------------------------
'''
def incomes_modelMoments_GA(alpha, beta):
    '''
    --------------------------------------------------------------------
    This function computes the 42 model moments for GMM
    --------------------------------------------------------------------
    INPUTS:
    alpha = a parameter for GA 
    beta  = a parameter for GA
 
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        sts.gamma.cdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    GA_cdf        = a lambda function of Gamma cdf 
    model_moments = a list of length 42 that contains model moments
    bins          = a list that contains the intervals of each bins in
                    the incomes histogram
    prob          = a model moment

    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: model_moments
    --------------------------------------------------------------------
    '''
    GA_cdf = lambda x: sts.gamma.cdf(x, alpha, 0, beta)
    model_moments = []
    # create a variable called bins
    bins = list(range(0,205,5)) + [250,350] 

    for index in range(len(bins)-1):
        prob = GA_cdf(bins[index + 1]) - \
               GA_cdf(bins[index])
        model_moments.append(prob)

    return model_moments


def incomes_err_vec_GA(df, alpha, beta, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    df = a pandas dataframe, incomes
    alpha = a parameter for GA 
    beta  = a parameter for GA
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        incomes_dataMoments()
        incomes_modelMoments_GA()
    
    OBJECTS CREATED WITHIN FUNCTION:
    data_moments  = a list of 42 data moments
    moms_data     = same as data_moments but in the form of np.array
    model_moments = a list of 42 model moments
    moms_model    = same as model_moments but in the form of np.array
    err_vec       = column vector of two moment error
                    functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    # compute data moments
    data_moments = incomes_dataMoments(df)
    moms_data = np.array(data_moments)

    # compute model moments
    model_moments = incomes_modelMoments_GA(alpha, beta)
    moms_model = np.array(model_moments)

    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data

    return err_vec


def incomes_criterion_GA(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([alpha, beta])
    alpha = a parameter for GA 
    beta  = a parameter for GA
    args   = length 2 tuple, (df, W)
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
 
    OBJECTS CREATED WITHIN FUNCTION:
    err        = column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    alpha, beta = params
    df, W = args
    err = incomes_err_vec_GA(df, alpha, beta, simple = False)
    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val


# initial guess of the lognormal parameters
alpha_init = 3
beta_init = 30
params_init_GA = np.array( [alpha_init, beta_init] )
# define weighting matrix to be an identity matrix as required
W_hat_GA = np.diag(incomes.percent, 0)
# try identity matrix
W_hat_I = np.eye(42)
gmm_args_GA = (incomes, W_hat_I)
# implement optimization algorithm
results_GA = opt.minimize(incomes_criterion_GA, params_init_GA, \
                          args = (gmm_args_GA), \
                          method = 'L-BFGS-B', \
                          bounds = ((1e-10, None), (1e-10, None)))

# variable assignment of MLE parameters
alpha_GMM, beta_GMM = results_GA.x
# compute value of GMM criterion function using MOM estimators
val_crit_GA = incomes_criterion_GA([alpha_GMM, beta_GMM],incomes, W_hat_GA)

# print out the results
print('----------------------------------------------------------------------')
print('Problem 1/c')
print('alpha_GMM_GA = ', alpha_GMM)
print('beta_GMM_GA = ', beta_GMM)
print('val of GMM criterion function using GMM estimators: ', val_crit_GA)
print('-----------------------------------')
print(results_GA)
print('----------------------------------------------------------------------')
print()

# simulate some points for distribution plotting
dist_vals_GA = incomes_modelMoments_GA(alpha_GMM, beta_GMM)
dist_vals_GA[40] /= 10
dist_vals_GA[41] /= 20

GMM_GA = True
if GMM_GA: 
    '''
    -------------------------------------------------------------------- 
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved 
    -------------------------------------------------------------------- 
    ''' 
    # initialite a plot
    fig, axes = plt.subplots()

    # create histogram
    n, bin_cuts, patches = plt.hist(incomes.midpoint, bins = bins, \
                                    weights=weights, color = 'maroon')
    # plot the Gamma distribution
    plt.plot(dist_pts, dist_vals_GA, \
             linewidth=2, color='blue', \
             label=r'$GA (\hat{\alpha}_{GMM}, \hat{\beta}_{GMM})$')
    plt.legend(loc='upper right')

    # add grid 
    minorLocator = MultipleLocator(1)
    axes.xaxis.set_minor_locator(minorLocator)  
    plt.grid(b=True, which='major', axis = 'both', color='0.65', linestyle='-')

    # set xlimit
    plt.xlim([0, 350])
    plt.ylim([0, 0.09])
    
    # adding plot title and xlabel and ylabel
    plt.title('Distribution of household incomes in the U.S.')    
    plt.xlabel(r'Household Income (in $1,000)')
    plt.ylabel(r'Percent of observations in bin')
    
    # generate plot in the designated directory
    output_path = os.path.join(output_dir, 'Fig_3')
    plt.savefig(output_path)   
    plt.close()  


'''
------------------------------------------------------------------------------
Exercise 1/d.: Plot the histogram from part (a) overlayed with the 
               distributions estimated in parts (b) and (c) 
------------------------------------------------------------------------------
LN_GA = Boolean, =True if make a plot of incomes distribution and the 
        probability density functions of lognormal and Gamma using MOM 
        parameters
------------------------------------------------------------------------------
'''
LN_GA = True
if LN_GA:
    '''
    -------------------------------------------------------------------- 
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved 
    -------------------------------------------------------------------- 
    ''' 
    # initialite a plot
    fig, axes = plt.subplots()

    # create histogram
    n, bin_cuts, patches = plt.hist(incomes.midpoint, bins = bins, \
                                    weights=weights, color = 'maroon')
    # plot the lognormal distribution
    plt.plot(dist_pts, dist_vals_LN, \
             linewidth=2, color='green', \
             label=r'$LN (\hat{\mu}_{GMM}, \hat{\sigma}_{GMM})$')
    # plot the Gamma distribution
    plt.plot(dist_pts, dist_vals_GA, \
             linewidth=2, color='blue', \
             label=r'$GA (\hat{\alpha}_{GMM}, \hat{\beta}_{GMM})$')
    plt.legend(loc='upper right')

    # add grid 
    minorLocator = MultipleLocator(1)
    axes.xaxis.set_minor_locator(minorLocator)  
    plt.grid(b=True, which='major', axis = 'both', color='0.65', linestyle='-')

    # set xlimit
    plt.xlim([0, 350])
    
    # adding plot title and xlabel and ylabel
    plt.title('Distribution of household incomes in the U.S.')    
    plt.xlabel(r'Household Income (in $1,000)')
    plt.ylabel(r'Percent of observations in bin')
    
    # generate plot in the designated directory
    output_path = os.path.join(output_dir, 'Fig_4')
    plt.savefig(output_path)   
    plt.close()  


'''
------------------------------------------------------------------------------
Exercise 1/e.: Repeat estimations of the GA distribution from part (c) using
               two-step estimator for the optimal weighting matrix
------------------------------------------------------------------------------
err_GA2S          = moment error vector in percentage using GMM estimator 
VCV_GA2S          = variance-covariance matrix based on moment error vector
W_hat_GA2S        = two-step weighting matrix
params_init_GA2Si = initial parameters using GMM estimators
gmm_args_GA2S     = new optimization arguments with two-step weighting matrix
results_GA2S      = an optimization result object
alpha_GMM_2S      = GMM estimator for alpha
beta_GMM_2S       = GMM estimator for beta
val_crit_GA2S     = value of GMM criterion function using GMM estimators
dist_vals_GA2S    = values used for plotting estimated distribution
GMM_GA2S          = Boolean, =True if make a plot of incomes distribution 
                    and the probability density function of lognormal,
                    Gamma, and Gamma 2-steps distributions using GMM parameters
------------------------------------------------------------------------------
'''
# compute the vector of moment errors in percentage using GMM estimators
err_GA2S = incomes_err_vec_GA(incomes, alpha_GMM, beta_GMM, False)
err_GA2S = np.array(err_GA2S).reshape(42, 1)
# computes the variance-covariance matrix based on moment errors
VCV_GA2S = np.dot(err_GA2S, err_GA2S.T) / incomes.shape[0]
# take the inverse of the vcv matrix so as to get the two-step weighting matrix
W_hat_GA2S = lin.pinv(VCV_GA2S)

# initial values now turned out to be GMM estimators
params_init_GA2S = np.array( [alpha_GMM, beta_GMM] )
gmm_args_GA2S = (incomes, W_hat_GA2S)
results_GA2S = opt.minimize(incomes_criterion_GA, \
                            params_init_GA2S, \
                            args = (gmm_args_GA2S), \
                            method = 'L-BFGS-B', \
                            bounds = ((1e-12, None), (1e-12, None)))

# variable assignment of GMM parameters
alpha_GMM_2S, beta_GMM_2S = results_GA2S.x
# compute value of GMM criterion function using 2-steps GMM estimators
val_crit_GA2S = incomes_criterion_GA([alpha_GMM_2S, beta_GMM_2S], \
                                     incomes, W_hat_GA2S)

# print out the results
print('----------------------------------------------------------------------')
print('Problem 1/e')
print('alpha_GMM_GA2S = ', alpha_GMM_2S)
print('beta_GMM_GA2S = ', beta_GMM_2S)
print(r'val of GMM criterion function using GMM estimators and optimal vcov' \
      + ' matrix: ', val_crit_GA2S)
print('-----------------------------------')
print(results_GA2S)
print('----------------------------------------------------------------------')
print()


# simulate some points for distribution plotting
dist_vals_GA2S = incomes_modelMoments_GA(alpha_GMM_2S, beta_GMM_2S)
dist_vals_GA2S[40] /= 10
dist_vals_GA2S[41] /= 20


GMM_GA2S = True
if GMM_GA2S: 
    '''
    -------------------------------------------------------------------- 
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved 
    -------------------------------------------------------------------- 
    ''' 
    # initialite a plot
    fig, axes = plt.subplots()

    # create histogram
    n, bin_cuts, patches = plt.hist(incomes.midpoint, bins = bins, \
                                    weights=weights, color = 'maroon')
    # plot the lognormal distribution
    plt.plot(dist_pts, dist_vals_LN, \
             linewidth=2, color='green', \
             label=r'$LN (\hat{\mu}_{GMM}, \hat{\sigma}_{GMM})$')
    # plot the Gamma distribution
    plt.plot(dist_pts, dist_vals_GA, \
             linewidth=2, color='blue', \
             label=r'$GA (\hat{\alpha}_{GMM}, \hat{\beta}_{GMM})$')
    # plot the Gamma 2-step
    plt.plot(dist_pts, dist_vals_GA2S, \
             linewidth=2, color='yellow', linestyle = ':', marker = '*', \
             label=r'$GA (\hat{\alpha}^{2Step}_{GMM}, ' + 
                   r'\hat{\beta}^{2Step}_{GMM})$')
    plt.legend(loc='upper right')

    # add grid 
    minorLocator = MultipleLocator(1)
    axes.xaxis.set_minor_locator(minorLocator)  
    plt.grid(b=True, which='major', axis = 'both', color='0.65', linestyle='-')

    # set xlimit
    plt.xlim([0, 350])
    plt.ylim([0, 0.09])
    
    # adding plot title and xlabel and ylabel
    plt.title('Distribution of household incomes in the U.S.')    
    plt.xlabel(r'Household Income (in $1,000)')
    plt.ylabel(r'Percent of observations in bin')
    
    # generate plot in the designated directory
    output_path = os.path.join(output_dir, 'Fig_5')
    plt.savefig(output_path)   
    plt.close()  
