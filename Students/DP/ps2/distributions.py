# MACS 40200: PS2
# Name: Dongping Zhang
# Python Version: 3.5
# Seed: None

import numpy as np
import scipy.special as spc
import scipy.stats as sts

def GA_pdf(xvals, alpha, beta):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the Gamma pdf with parameters alpha and
    beta
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of random varaible follow a Gamma 
             distribution
    alpha  = scalar > 0, a parameter for the Gamma distribution
    beta   = scalar > 0, a parameter for the Gamma distribution
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, Gamma PDF values for alpha and beta
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    # gamma function in the denominator of gamma distribution: gamma(n)=(n-1)!
    pdf_vals = ( 1 / (beta**alpha * spc.gamma(alpha)) ) * \
               ( xvals**(alpha - 1) ) * \
               ( np.exp(-xvals / beta) )  

    return pdf_vals


def GG_pdf(xvals, alpha, beta, m):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the Generalized Gamma pdf with parameters 
    alpha, beta. and m
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of random varaible follow a 
             Generalized Gamma distribution
    alpha  = scalar > 0, a parameter for the Generalized Gamma distribution
    beta   = scalar > 0, a parameter for the Generalized Gamma distribution
    m      = scalar > 0, a parameter for the Generalized Gamma distribution
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, Generalized Gamma PDF values for alpha, beta
               and m corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    # gamma function in the denominator of gamma distribution: gamma(n)=(n-1)!
    pdf_vals = ( ((m / beta**alpha) / spc.gamma(alpha / m)) * \
                 (xvals**(alpha - 1)) * \
                 np.exp( -(xvals / beta)**m) ) 
    return pdf_vals


def GB2_pdf(xvals, a, b, p, q):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the Generalized Beta 2 pdf with parameters 
    a, b, p, and q
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of random varaible follow a 
             Generalized Beta 2 distribution
    a      = scalar > 0, a parameter for the Generalized Beta 2 distribution
    b      = scalar > 0, a parameter for the Generalized Beta 2 distribution
    p      = scalar > 0, a parameter for the Generalized Beta 2 distribution
    q      = scalar > 0, a parameter for the Generalized Beta 2 distribution
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, Generalized Beta 2 PDF values for a, b, p, and
               q corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    pdf_vals = ( np.abs(a) * xvals**(a * p - 1) ) / \
               ( b**(a * p) * spc.beta(p, q) * (1 + (xvals / b)**a)**(p+q) ) 
    return pdf_vals
