'''
------------------------------------------------------------------------
This is Bobae's main script for PS3 of MACS 40000: Structural Estimation.
------------------------------------------------------------------------
'''
# import pacakges
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import scipy as sp
import scipy.stats as sts
import scipy.optimize as opt
import matplotlib.pyplot as plt

# import data
data = DataFrame(pd.read_csv('MacroSeries.txt', header = None))
ct, kt, wt, rt = data[0], data[1], data[2], data[3]
'''
(a) Use the data (wt, kt) and equations (3) and (5) to estimate the
four parameters (alpha, rho, mu, sigma) by maximum likelihood.
'''
def a_pdf(wt, kt, params):

    alpha, rho, mu, sigma = params[0], params[1], params[2], params[3]

    zt = np.log( wt / (1.0 - rho) / kt**alpha )
    # zt = np.log(wt)- np.log(1.0 - rho) - alpha* np.log(kt)
    zt_lag = []
    for i in range(len(zt)):
        if i == 0:
            zt_lag.append(mu)
        else:
            zt_lag.append(zt[i - 1])

    loc0 = (rho * Series(zt_lag)) + ((1.0 - rho) * mu)
    pdf_vals = sts.norm.pdf(Series(zt), loc = loc0, scale = sigma)

    return pdf_vals

def log_lik_a(wt, kt, params):

    pdf_vals = a_pdf(wt, kt, params)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val

def crit_a(params, *args):

    wt, kt = args
    log_lik_val = log_lik_a(wt, kt, params)
    neg_log_lik_val = -log_lik_val

    return neg_log_lik_val

# initial attempt
a_mle_args = (wt, kt)
a_bounds = ((1e-10, 1 - 1e-10), (-1 + 1e-10, 1 - 1e-10), (1e-10, None), (1e-10, None))
a_alpha_0, a_rho_0, a_mu_0, a_sig_0 = 0.5, 0.5, 5.0, 5.0
a_params_1 = np.array([a_alpha_0, a_rho_0, a_mu_0, a_sig_0])
a_results1 = opt.minimize(crit_a, a_params_1, args=(a_mle_args), bounds = a_bounds,
                        method = 'SLSQP', options ={'ftol': 1e-10}) # other methods, 'L-BFGS-B', 'SLSQP'
a_alpha_MLE1, a_rho_MLE1, a_mu_MLE1, a_sig_MLE1 = a_results1.x
# print(a_results1) # optimization succeeded!

# second attempt with first MLE values (to get a hessian matrix)
a_params_2 = np.array([a_alpha_MLE1, a_rho_MLE1, a_mu_MLE1, a_sig_MLE1])
a_results2 = opt.minimize(crit_a, a_params_2, args=(a_mle_args), bounds = a_bounds,
                        method = 'L-BFGS-B', options ={'ftol': 1e-10}) # other methods, 'L-BFGS-B', 'SLSQP'
a_alpha_MLE2, a_rho_MLE2, a_mu_MLE2, a_sig_MLE2 = a_results2.x
# print(a_results2) # optimization succeeded!

if True: # report the results
    print('Report the MLE results for (a):')
    print('alpha_MLE = ', a_alpha_MLE2, 'rho_MLE = ', a_rho_MLE2,
        'mu_MLE = ', a_mu_MLE2, ' sig_MLE = ', a_sig_MLE2)
    print('the value of the likelihood function = ', a_results2.fun)
    print('VCV = ', a_results2.hess_inv.todense())

'''
(b) Use the data (rt, kt) and equations (4) and (5) to estimate the
four parameters (alpha, rho, mu, sigma) by maximum likelihood.
'''
def b_pdf(rt, kt, params):

    alpha, rho, mu, sigma = params[0], params[1], params[2], params[3]

    zt = np.log(rt / (alpha * (kt**alpha)))
    # zt = np.log(rt) - np.log(alpha) - alpha*np.log(kt)

    zt_lag = []
    for i in range(len(zt)):
        if i == 0:
            zt_lag.append(mu)
        else:
            zt_lag.append(zt[i - 1])

    loc0 = (rho * Series(zt_lag)) + ((1.0 - rho) * mu)
    pdf_vals = sts.norm.pdf(Series(zt), loc = loc0, scale = sigma)

    return pdf_vals

def log_lik_b(rt, kt, params):

    pdf_vals = a_pdf(rt, kt, params)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val

def crit_b(params, *args):

    rt, kt = args
    log_lik_val = log_lik_a(rt, kt, params)
    neg_log_lik_val = -log_lik_val

    return neg_log_lik_val

# initial attempt, using the MLE results from (a)
b_mle_args = (rt, kt)
b_bounds = ((1e-10, 1 - 1e-10), (-1 + 1e-10, 1 - 1e-10), (1e-10, None), (1e-10, None))
b_alpha_0, b_rho_0, b_mu_0, b_sig_0 = 0.5, 0.5, 10.0, 1.0
b_params_1 = np.array([b_alpha_0, b_rho_0, b_mu_0, b_sig_0])
b_results1 = opt.minimize(crit_b, b_params_1, args=(b_mle_args), bounds = b_bounds,
                        method = 'L-BFGS-B', options ={'ftol': 1e-10}) # other methods, 'L-BFGS-B', 'SLSQP'
b_alpha_MLE1, b_rho_MLE1, b_mu_MLE1, b_sig_MLE1 = b_results1.x
# print(b_results1) # optimization succeeded!

# second attempt using the first MLE values
b_params_2 = np.array([b_alpha_MLE1, b_rho_MLE1, b_mu_MLE1, b_sig_MLE1])
b_results2 = opt.minimize(crit_b, b_params_2, args=(b_mle_args), bounds = b_bounds,
                        method = 'L-BFGS-B', options ={'ftol': 1e-10}) # other methods, 'L-BFGS-B', 'SLSQS'
b_alpha_MLE2, b_rho_MLE2, b_mu_MLE2, b_sig_MLE2 = b_results2.x
# print(b_results2)  # optimization succeeded with smaller score!

# third attempt using the second MLE values
b_params_3 = np.array([b_alpha_MLE2, b_rho_MLE2, b_mu_MLE2, b_sig_MLE2])
b_results3 = opt.minimize(crit_b, b_params_3, args=(b_mle_args), bounds = b_bounds,
                        method = 'L-BFGS-B', options ={'ftol': 1e-10}) # other methods, 'L-BFGS-B', 'SLSQS'
b_alpha_MLE3, b_rho_MLE3, b_mu_MLE3, b_sig_MLE3 = b_results3.x
# print(b_results3) # optimization succeeded with smaller score! (another attempt yields the same score)

if True: # report the result
    print('Report the MLE results for (b):')
    print('alpha_MLE = ', b_alpha_MLE3, 'rho_MLE = ', b_rho_MLE3,
        'mu_MLE = ', b_mu_MLE3, ' sig_MLE = ', b_sig_MLE3)
    print('the value of the likelihood function = ', b_results3.fun)
    print('VCV = ', b_results3.hess_inv.todense())

'''
(c) According to your estimates from part (a), if investment/savings
in the current period is kt = 7,500,000 and the productivity shock in the
previous period was zt_lag = 10, what is the probability that the interest
rate this period will be greater than rt = 1?
'''
z_star = np.log(1 / (a_alpha_MLE2 * (7500000)**a_alpha_MLE2))
loc_c = (a_rho_MLE2 * Series(10)) + ((1.0 - a_rho_MLE2) * a_mu_MLE2)
c_prob = 1 - sts.norm.cdf(z_star, loc = loc_c, scale = a_sig_MLE2)
if True: # report the result
    print('Pr(rt > 1 | theta-hat, kt = 7500000, z_lag = 10) = ', c_prob)
