# MACS 40200: PS2
# Name: Dongping Zhang
# Python Version: 3.5
# Seed: None

import pandas as pd
import numpy as np
import scipy.stats as sts
import os
import matplotlib.pyplot as plt
import distributions as dst
import scipy.special as spc
import scipy.stats as sts
import scipy.optimize as opt
import scipy.integrate as integration
import math

'''
------------------------------------------------------------------------------
Exercise 1/a.: Plot a density histogram with incomes data
------------------------------------------------------------------------------
clms        = the raw dataset of the health claim amounts
clms_stat   = some descriptive statistics of clms
clms_median = the median of the clms dataset
clms_std    = the standard deviation of the clms dataset
clms_raw    = Boolean, =True if make a plot of the density histogram of clms
clms_clean  = Boolean, =True if make a plot of the density histogram of all 
              clms data points that are less of equal to 800
------------------------------------------------------------------------------
'''
# load health claims data from the raw txt file
clms= np.loadtxt('clms.txt')

# summary statistics: mean, median, maximum, minimum, and sd
clms_stat = sts.describe(clms)
# calculate median
clms_median = np.median(clms)
# calculate standard deviation
clms_std = np.std(clms)

# print out the result
print('mean = ' + str(clms_stat.mean))
print('median = ' + str(clms_median))
print('maximum = ' + str(clms_stat.minmax[1]))
print('minimum = ' + str(clms_stat.minmax[0]))
print('variance = ' + str(clms_stat.variance))
print('standard deviation = ' + str(clms_std))


clms_raw = True
if clms_raw:
    '''
    -------------------------------------------------------------------- 
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved 
    -------------------------------------------------------------------- 
    ''' 
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
  
    # create a subplot
    fig, ax = plt.subplots()

    # create histogram
    count, bins, ignored = plt.hist(clms, 1000, normed=True, color='maroon')

    # create plot title and xlabel and ylabel
    plt.title('Histogram of Raw Households Health Expenditure')
    plt.xlabel(r'Health Expenditure ($\$$)')
    plt.ylabel(r'Percent of Observations in bin')    
    
    # output of the plot into the image directory
    output_path = os.path.join(output_dir, 'Fig_1')
    plt.savefig(output_path) 
    plt.close() 


clms_clean = True
if clms_clean:
    '''
    -------------------------------------------------------------------- 
    bin_numi = the number of bins used in the plot
    -------------------------------------------------------------------- 
    ''' 
    # find the values that are <= 800
    clms_truncated = clms[clms <= 800]
    # create a subplot
    fig, ax = plt.subplots()

    # density histogram weights creation
    weights = ( len(clms_truncated) / len(clms) / len(clms_truncated) ) * \
              np.ones_like(clms_truncated)

    # determine the number of bins needed
    bin_num = 100

    # create histogram
    n, bin_cuts, patches = plt.hist(clms_truncated, bin_num, \
                                    weights = weights,\
                                    color = 'maroon')

    # create plot title and xlabel and ylabel
    plt.title('Histogram of Truncated Households Health Expenditure')
    plt.xlabel(r'Health Expenditure ($\$$)')
    plt.ylabel(r'Percent of Observations in bin')    
   
    # output of the plot into the image directory
    output_path = os.path.join(output_dir, 'Fig_2')
    plt.savefig(output_path) 
    plt.close() 

# to show that the sun of all bins in clms_clean plot is not 1
print('Sum of prob represented by all bins: ' + str(np.sum(n)))


'''
------------------------------------------------------------------------------
Exercise 1/b: fit the gamma distribution to the individual observation data
------------------------------------------------------------------------------
log_lik_GA  = a function used to compute the log likelihood value of the 
              Gamma distribution given parameters alpha and beta
crit_GA     = the objective function of the Gama distribution
dist_pts    = simulated points to generate plots
beta_0      = initial guess of the beta parameter to the Gamma distribution
alpha_0     = initial guess of the alpha parameter to the Gamma distribution
alpha_init  = the same as alpha_0, used to clarify the optimization algorithm
beta_init   = the same as beta_0, used to clarify  the optimization algorithm
params_init = an array of the parameters of the Gamma distribution
mle_args    = the MLE arguments or the data points
GA_results  = the optimization result for the Gamma distribution
alpha_MLE   = the MLE estimator for the alpha parameter
beta_MLE    = the MLE estimator for the beta parameter
clms_GA_MLE = Boolean, =True if make a plot of the density histogram of clms
              and overlap the Gamma distribution to the histogram 
------------------------------------------------------------------------------
'''
# log-likelihood function for the GA distribution
def log_lik_GA(xvals, alpha, beta):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given Gamma
    distribution parameters alpha and beta.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of random varaible follow a Gamma 
             distribution
    alpha  = scalar > 0, a parameter for the Gamma distribution
    beta   = scalar > 0, a parameter for the Gamma distribution
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    GA_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, Gamma PDF values for alpha and beta
                  corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of Gamma PDF values
                  for alpha and beta corresponding to xvals data
    log_lik_val = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    pdf_vals = dst.GA_pdf(xvals, alpha, beta)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    
    return log_lik_val


# define the objective function
def crit_GA(params, args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([alpha, beta])
    alpha  = scalar > 0, a parameter for the Gamma distribution
    beta   = scalar > 0, a parameter for the Gamma distribution
    args   = length 1 tuple, (xvals)
    xvals  = (N,) vector, values of random variable follows a Gamma
             distribution
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    log_lik_GA()
    
    OBJECTS CREATED WITHIN FUNCTION:
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of log_lik_val
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    alpha, beta = params
    xvals = args
    log_lik_val = log_lik_GA(xvals, alpha, beta)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val


# simulate points for plotting
dist_pts = np.linspace(0, 800, 800)

# initial guess of the gama parameters
beta_0 = np.var(clms) / np.mean(clms)
alpha_0 = np.mean(clms) / beta_0 

# compute the MLE parameter estimation
alpha_init = alpha_0 
beta_init = beta_0 
params_init = np.array([alpha_init, beta_init])
mle_args = (clms)
GA_results = opt.minimize(crit_GA, params_init, args=(mle_args), \
                          bounds = ((0, math.inf), (0, math.inf)))
alpha_MLE, beta_MLE = GA_results.x
print('GA: alpha_MLE: ', alpha_MLE)
print('GA: beta_MLE: ', beta_MLE)
print('Max log likelihood for GA: ', log_lik_GA(clms, alpha_MLE, beta_MLE))


clms_GA_MLE = True
if clms_GA_MLE: 
    '''
    -------------------------------------------------------------------- 
    bin_numi = the number of bins used in the plot
    -------------------------------------------------------------------- 
    ''' 
    # create a subplot
    fig, ax = plt.subplots()

    # density histogram weights creation
    weights = (1 / clms.shape[0]) * np.ones_like(clms)

    # determine the number of bins needed
    bin_num = 100

    # create histogram
    n, bin_cuts, patches = plt.hist(clms, bin_num, \
                                    weights = weights,\
                                    color = 'maroon', \
                                    range = ([0, 800]))

    # gamma density MLE
    plt.plot(dist_pts, \
             dst.GA_pdf(dist_pts, alpha_MLE, beta_MLE), \
             linewidth = 3, color = 'green', \
             label = r'GA MLE: $\alpha$ = ' + str(np.round(alpha_MLE, 2)) + \
                     r', $\beta$ = ' + str(np.round(beta_MLE, 2)))

    # plt legend
    plt.legend(loc = 'upper right')

    # set the limits of axis
    plt.xlim([0, 800])
    plt.ylim([0, 0.045])

    # create plot title and xlabel and ylabel
    plt.title('Histogram of Truncated Households Health Expenditure')
    plt.xlabel(r'Health Expenditure ($\$$)')
    plt.ylabel(r'Percent of Observations in bin')     

    # output of the plot into the image directory
    output_path = os.path.join(output_dir, 'Fig_3')
    plt.savefig(output_path) 
    plt.close() 


'''
------------------------------------------------------------------------------
Exercise 1/c: fit the genralized gamma distribution to the individual 
              observation data
------------------------------------------------------------------------------
log_lik_GG     = a function used to compute the log likelihood value of the 
                 Generalized Gamma distribution given parameters alpha,  beta
                 and m
crit_GG        = the objective function of the Generalized Gamma distribution
alpha_init_GG  = initial guess of the alpha parameter 
beta_init_GG   = initial guess of the beta parameter
m_init_GG      = initial guess of the m parameter
params_init_GG = an array of the parameters of the Generalized Gamma 
                 distribution
mle_args_GG    = the MLE arguments or the data points
GG_results     = the optimization result for the Generalized Gamma distribution
alpha_MLE_GG   = the MLE estimator of alpha parameter
beta_MLE_GG    = the MLE estimator of beta parameter
m_MLE_GG       = the MLE estimator of m parameter
clms_GG_MLE    = Boolean, =True if make a plot of the density histogram of clms
                 and overlap the Generalized Gamma distribution to the 
                 histogram  
------------------------------------------------------------------------------
'''
# log likelihood for Genralized  GA distribution
def log_lik_GG(xvals, alpha, beta, m):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given the 
    Generalized Gamma distribution parameters alpha, beta, and m.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of random varaible follow a 
             Generalized Gamma distribution
    alpha  = scalar > 0, a parameter for the Generalized Gamma distribution
    beta   = scalar > 0, a parameter for the Generalized Gamma distribution
    m      = scalar > 0, a parameter for the Generalized Gamma distribution

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    GG_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, Generalized Gamma PDF values for 
                  alpha, beta, and m corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of the Generalized  
                  Gamma PDF values for alpha, beta, and m corresponding 
                  to xvals data
    log_lik_val = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    pdf_vals = dst.GG_pdf(xvals, alpha, beta, m)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    
    return log_lik_val


# define the objective function
def crit_GG(params, args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (3,) vector, ([alpha, beta, m])
    alpha  = scalar > 0, a parameter for the Generalized Gamma distribution
    beta   = scalar > 0, a parameter for the Generalized Gamma distribution
    m      = scalar > 0, a parameter for the Generalized Gamma distribution
    args   = length 1 tuple, (xvals)
    xvals  = (N,) vector, values of random variable follows a Generalized 
             Gamma distribution
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    log_lik_GG()
    
    OBJECTS CREATED WITHIN FUNCTION:
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of log_lik_val
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    alpha, beta, m = params
    xvals = args
    log_lik_val = log_lik_GG(xvals, alpha, beta, m)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val


# compute the MLE parameter estimation
alpha_init_GG = alpha_MLE
beta_init_GG = beta_MLE  
m_init_GG = 1
params_init_GG = np.array([alpha_init_GG, beta_init_GG, m_init_GG])
mle_args_GG = (clms)
GG_results = opt.minimize(crit_GG, params_init_GG, args=(mle_args_GG), \
                          bounds = ((0, math.inf), \
                                    (0, math.inf), \
                                    (0, math.inf)))
alpha_MLE_GG, beta_MLE_GG, m_MLE_GG = GG_results.x
print('GG: alpha_MLE: ', alpha_MLE_GG)
print('GG: beta_MLE: ', beta_MLE_GG)
print('GG: m_MLE: ', m_MLE_GG)
print('Max Log-likelihood for GG:', log_lik_GG(clms, alpha_MLE_GG, beta_MLE_GG, \
                                           m_MLE_GG))


clms_GG_MLE = True
if clms_GG_MLE: 
    '''
    -------------------------------------------------------------------- 
    bin_numi = the number of bins used in the plot
    -------------------------------------------------------------------- 
    ''' 
    # create a subplot
    fig, ax = plt.subplots()

    # density histogram weights creation
    weights = (1 / clms.shape[0]) * np.ones_like(clms)

    # determine the number of bins needed
    bin_num = 100

    # create histogram
    n, bin_cuts, patches = plt.hist(clms, bin_num, \
                                    weights = weights,\
                                    color = 'maroon', \
                                    range = ([0, 800]))

    # genralized gamma density MLE
    plt.plot(dist_pts, \
             dst.GG_pdf(dist_pts, alpha_MLE_GG, beta_MLE_GG, m_MLE_GG), \
             linewidth = 3, color = 'blue', \
             label = r'GG MLE: ' + \
                     r'$\alpha$ = ' + str(np.round(alpha_MLE_GG, 2)) + \
                     r', $\beta$ = ' + str(np.round(beta_MLE_GG, 2)) + \
                     r'. m = ' + str(np.round(m_MLE_GG, 2)))

    # plt legend
    plt.legend(loc = 'upper right')

    # set the limits of axis
    plt.xlim([0, 800])
    plt.ylim([0, 0.045])

    # create plot title and xlabel and ylabel
    plt.title('Histogram of Truncated Households Health Expenditure')
    plt.xlabel(r'Health Expenditure ($\$$)')
    plt.ylabel(r'Percent of Observations in bin')     

    # output of the plot into the image directory
    output_path = os.path.join(output_dir, 'Fig_4')
    plt.savefig(output_path) 
    plt.close() 


'''
------------------------------------------------------------------------------
Exercise 1/d: fit the genralized beta2 distribution to the individual 
              observation data
------------------------------------------------------------------------------
log_lik_GB2     = a function used to compute the log likelihood value of the 
                  Generalized Beta 2 distribution given parameters a, b, p, q
crit_GB2        = the objective function of the Generalized Beta 2 distribution
a_init_GB2      = initial guess of the a parameter
q_init_GB2      = initial guess of the q parameter
b_init_GB2      = initial guess of the b parameter
p_init_GB2      = initial guess of the p parameter
params_init_GB2 = an array of the parameters of the Generalized Beta 2 
                  distribution
mle_args_GB2    = the MLE arguments or the data points
GB2_results     = the optimization result for the Generalized Beta 2 
                  distribution
a_MLE_GB2       = the MLE estimator of a parameter
b_MLE_GB2       = the MLE estimator of b parameter 
p_MLE_GB2       = the MLE estimator of p parameter 
q_MLE_GB2       = the MLE estimator of q parameter 
clms_GB2_MLE    = Boolean, =True if make a plot of the density histogram of 
                  clms and overlap the Generalized Beta 2 distribution to the 
                  histogram
------------------------------------------------------------------------------
'''
# log likelihood for beta2 distribution
def log_lik_GB2(xvals, a, b, p, q):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given the 
    Generalized Beta 2 distribution parameters a, b, p, q.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of random varaible follow a 
             Generalized Beta 2 distribution
    a      = scalar > 0, a parameter for the Generalized Beta 2 distribution
    b      = scalar > 0, a parameter for the Generalized Beta 2 distribution
    p      = scalar > 0, a parameter for the Generalized Beta 2 distribution
    q      = scalar > 0, a parameter for the Generalized Beta 2 distribution

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    GB2_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, Generalized Beta 2 PDF values for 
                  a, b, p, and q corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of the Generalized  
                  Beta 2 PDF values for a, b, p, and q corresponding 
                  to xvals data
    log_lik_val = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    pdf_vals = dst.GB2_pdf(xvals, a, b, p, q)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    
    return log_lik_val


# define the objective function
def crit_GB2(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (4,) vector, ([a, b, p, q])
    a      = scalar > 0, a parameter for the Generalized Beta 2 distribution
    b      = scalar > 0, a parameter for the Generalized Beta 2 distribution
    p      = scalar > 0, a parameter for the Generalized Beta 2 distribution
    q      = scalar > 0, a parameter for the Generalized Beta 2 distribution
    args   = length 1 tuple, (xvals)
    xvals  = (N,) vector, values of random variable follows a Generalized 
             Beta 2 distribution
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    log_lik_GB2()
    
    OBJECTS CREATED WITHIN FUNCTION:
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of log_lik_val
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    a, b, p, q = params
    xvals = args
    log_lik_val = log_lik_GB2(xvals, a, b, p, q)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val


# compute the MLE parameter estimation
a_init_GB2 = m_MLE_GG
q_init_GB2 = 10000
b_init_GB2 = q_init_GB2**(1 / m_MLE_GG) * beta_MLE_GG
p_init_GB2 = alpha_MLE_GG / m_MLE_GG
params_init_GB2 = np.array([a_init_GB2, b_init_GB2, p_init_GB2, q_init_GB2])
mle_args_GB2 = (clms)
GB2_results = opt.minimize( crit_GB2, params_init_GB2, args=(mle_args_GB2), \
                            bounds = ( (0, math.inf), (0, math.inf), \
                                       (0, math.inf), (0, math.inf)) )
a_MLE_GB2, b_MLE_GB2, p_MLE_GB2, q_MLE_GB2 = GB2_results.x
print('GB2: a_MLE: ', a_MLE_GB2)
print('GB2: b_MLE: ', b_MLE_GB2)
print('GB2: p_MLE: ', p_MLE_GB2)
print('GB2: q_MLE: ', q_MLE_GB2)
print('Log-likelihood for GB2:', log_lik_GB2(clms, \
                                             a_MLE_GB2, b_MLE_GB2, \
                                             p_MLE_GB2, q_MLE_GB2))


clms_GB2_MLE = True
if clms_GB2_MLE: 
    '''
    -------------------------------------------------------------------- 
    bin_numi = the number of bins used in the plot
    -------------------------------------------------------------------- 
    ''' 
    # create a subplot
    fig, ax = plt.subplots()

    # density histogram weights creation
    weights = (1 / clms.shape[0]) * np.ones_like(clms)

    # determine the number of bins needed
    bin_num = 100

    # create histogram
    n, bin_cuts, patches = plt.hist(clms, bin_num, \
                                    weights = weights,\
                                    color = 'maroon', \
                                    range = ([0, 800]))

    # genralized gamma density MLE
    plt.plot(dist_pts, \
             dst.GB2_pdf(dist_pts, a_MLE_GB2, b_MLE_GB2, \
                                   p_MLE_GB2, q_MLE_GB2), \
             linewidth = 3, color = 'gold', \
             label = r'GB2 MLE: ' + \
                     '\n a = ' +str(np.round(a_MLE_GB2, 2)) + \
                     r', b = ' + str(np.round(b_MLE_GB2, 2)) + \
                     ', \n p = ' + str(np.round(p_MLE_GB2, 2)) + \
                     r', q = ' + str(np.round(q_MLE_GB2, 2)))

    # plt legend
    plt.legend(loc = 'upper right')

    # set the limits of axis
    plt.xlim([0, 800])
    plt.ylim([0, 0.045])

    # create plot title and xlabel and ylabel
    plt.title('Histogram of Truncated Households Health Expenditure')
    plt.xlabel(r'Health Expenditure ($\$$)')
    plt.ylabel(r'Percent of Observations in bin')     

    # output of the plot into the image directory
    output_path = os.path.join(output_dir, 'Fig_5')
    plt.savefig(output_path) 
    plt.close() 


'''
------------------------------------------------------------------------------
Exercise 1/e: perform a likelihood ratio test for each of the estimated in 
              parts (b) and (c), respectively, against the GB2 specification    
              in part (d)
------------------------------------------------------------------------------
log_lik_GA_MLE   = the log likelihood value after fitting the data to the 
                   Gamma distribution using the estimated MLE parameters
log_lik_GG_MLE   = the log likelihood value after fitting the data to the
                   Generalized Gamma distribution using the estimated MLE
                   parameters
log_lik_GB2_MLE  = the log likelihood value after fitting the data to the 
                   Generalized Beta 2 distribution using the estimated MLE
                   parameters
LR_val_GA_on_GB2 = the likelihood ratio test statistics computed for the 
                   estimated Gamma against the estimated Generalized Beta 2 
pval_GA_on_GB2   = the pvalue for the likelihood ratio test statistics 
                   computed for the estimated Gamma against the estimated 
                   Generalized Beta 2 following a chi squared distribution
                   with dof = 4
LR_val_GG_on_GB2 = the likelihood ratio test statistics computed for the 
                   estimated Generalized Gamma against the estimated 
                   Generalized Beta 2 
pval_GG_on_GB2   = the pvalue for the likelihood ratio test statistics 
                   computed for the estimated Generalized Gamma against 
                   the estimated Generalized Beta 2 following a chi squared 
                   distribution with dof = 4 
------------------------------------------------------------------------------
'''
# compute the log likelihood values using estimated MLE parameters
log_lik_GA_MLE = log_lik_GA(clms, alpha_MLE, beta_MLE)
log_lik_GG_MLE = log_lik_GG(clms, alpha_MLE_GG,  beta_MLE_GG, m_MLE_GG) 
log_lik_GB2_MLE = log_lik_GB2(clms, a_MLE_GB2, b_MLE_GB2, p_MLE_GB2, q_MLE_GB2) 

# case 1: GA against GB2
LR_val_GA_on_GB2 = 2 * (log_lik_GA_MLE - log_lik_GB2_MLE)
print('The log likelihood test statistics of the estimated Gamma distribution \
against the Generalized Beta 2 distribution is: ', LR_val_GA_on_GB2)
pval_GA_on_GB2 = 1.0 - sts.chi2.cdf(LR_val_GA_on_GB2, 4)
print('chi squared of H0 with 4 degrees of freedom p-value = ', \
      pval_GA_on_GB2)

# case 2: GG against GB2
LR_val_GG_on_GB2 = 2 * (log_lik_GG_MLE - log_lik_GB2_MLE)
print('The log likelihood test statistics of the Generalized Gamma \
distribution against the Generalized Beta 2 distribution is: ', \
      LR_val_GG_on_GB2)
pval_GG_on_GB2 = 1.0 - sts.chi2.cdf(LR_val_GG_on_GB2, 4)
print('chi squared of H0 with 4 degrees of freedom p-value = ', \
      pval_GG_on_GB2)


'''
------------------------------------------------------------------------------
Exercise 1/f: using the estimated GB2 distribution and the estimated GA
              distribution to estimate the probability of observing data >1000
------------------------------------------------------------------------------
GB2_prob_g1000 = the probability of observing a monthly health care claim of
                 more than $1,000 under the estimated Generalized Beta 2 
                 distribution using MLE parameters
GA_prob_g1000  = the probability of observing a monthly health care claim of
                 more than $1,000 under the estimated Gamma distribution using
                 MLE parameters
------------------------------------------------------------------------------
'''
# probability of observing data > 1000 under GB2 distribution
GB2_prob_g1000 = 1 - integration.quad(lambda xvals: \
                                      dst.GB2_pdf(xvals, a_MLE_GB2, b_MLE_GB2, \
                                                 p_MLE_GB2, q_MLE_GB2), \
                                      0, 1000)[0]
print('The estimated probability of observing a monthly health care expense \
of more than $1,000 following the estimated Genralized Beta 2 distribution \
is: ', GB2_prob_g1000)

# probability of observing data > 1000 under GA distribution
GA_prob_g1000 = 1 - integration.quad(lambda xvals: \
                                     dst.GA_pdf(xvals, alpha_MLE, beta_MLE), \
                                     0, 1000)[0] 
print('The estimated probability of observing a monthly health care expense \
of more than $1,000 following the estimated Gama distribution is: ', \
GA_prob_g1000)
