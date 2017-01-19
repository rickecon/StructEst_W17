'''
------------------------------------------------------------------------
This is Bobae's supplementary script for MACS 40000: Structural Estimation.
The current script allows the main script to run smoothly.
------------------------------------------------------------------------
'''
# import packages
import numpy as np
import scipy.stats as sts
import scipy.special as spc

# pdf valus for the gamma distribution
def ga_pdf(xvals, alpha, beta):
    '''
    --------------------------------------------------------------------
    This function computes the gamma (GA) distribution values for given
    data and parameters
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of data, 0 <= xvals < inf
    alpha  = scalar, > 0
    beta   = scalar, > 0

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        sts.gamma.pdf()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf = vector, values of the gamma distribution

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf
    -------------------------------------------------------------------
    '''
    pdf_vals = sts.gamma.pdf(xvals, alpha, loc=0, scale=beta)

    return pdf_vals

# pdf valus for the generalized gamma distribution
def gg_pdf(xvals, alpha, beta, mm):
    '''
    --------------------------------------------------------------------
    This function computes the generalized gamma (GG) distribution value for
    given data and parameters
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of data, 0 <= xvals < inf
    alpha  = scalar, > 0
    beta   = scalar, > 0
    mm     = scalar, > 0

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        sts.gengamma.pdf()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf = vector, values of the generalized gamma distribution

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf
    --------------------------------------------------------------------
    '''
    pdf_vals = sts.gengamma.pdf(xvals, alpha, mm, loc=0, scale=beta)

    return pdf_vals

# pdf valus for the generalized beta 2 distribution
def gb2_pdf(xvals, aa, bb, pp, qq):
    '''
    --------------------------------------------------------------------
    This function computes the generalized beta 2 (GB2) distribution value
    for given data and parameters
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of data, 0 <= xvals < inf
    aa     = scalar, > 0
    bb     = scalar, > 0
    pp     = scalar, > 0
    qq     = scalar, > 0

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        spc.beta()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf = vector, values of the generalized beta 2 distribution

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf
    --------------------------------------------------------------------
    '''
    pdf_vals = ((aa * xvals**(aa * pp - 1)) /
            (bb**(aa * pp) * (1 + (xvals / bb) ** aa)**(pp + qq) * spc.beta(pp, qq)))

    return pdf_vals
