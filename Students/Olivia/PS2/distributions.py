### This is a module that evaluates the PDF 
### For either a scalar or vector
### For 3 different parameterizations from the Generalized Beta Family of Distributions

import scipy.special as spc
import scipy.stats as sts
import numpy as np 

def log_lik_gamma(xvals,alpha,beta):
    # This is the parameterization of the gamma distribution with alpha as shape and beta as rate
    # It takes a scalar or vector of x values
    # it returns a scalar log likelihood sum of the data
    ln_pdf_vals = sts.gamma.logpdf(xvals,alpha,loc=0,scale=beta)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val

def log_lik_gen_gamma(xvals,alpha,beta,mm):
    #alpha = np.exp(l1)
    #beta= np.exp(l2)
    #mm = np.exp(l3)
    # This is the generalized gamma distribution
    # all parameters positive
    log_lik_val = sts.gengamma.logpdf(xvals,alpha,mm,loc=0,scale=beta).sum()
    return log_lik_val

def log_lik_gen_beta_2(xvals,bb,pp,qq,aa):
    # a is unrestricted, rest of parameters positive
    pdf_vals = (np.abs(aa)*(xvals**(aa*pp -1)))/((bb**(aa*pp))*spc.beta(pp,qq)*((1+((xvals/bb)**aa))**(pp+qq)))
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val
