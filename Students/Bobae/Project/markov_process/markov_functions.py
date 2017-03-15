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
The current script defines the following functions:
    * data_moments()
    * erg_dist()
    * model_moments()
    * criterion()
---------------------------------------------------------------------------
'''


import numpy as np
import numpy.linalg as LA
import pandas as pd


'''
---------------------------------------------------------------------------
I. Define data_moments()
---------------------------------------------------------------------------
'''
# define functions for getting data moments
def data_moments(data, uncondPr = True):
    '''
    --------------------------------------------------------------------
    This function takes in a pandas dataframe of health expenditure and
    computes the data moments: 
    b_k: the unconditional probability that the current period and the 
         previous k periods all had positive expenditures.
    d_k: the unconditional probability that the current period and the 
         previous k periods all had zero expenditures. 

    In order to compute b_k and d_k, the function would first compute 
    a_k: the conditional probability that an individual had positive 
         expenditure in the current month given that the individual also
         had positive expenditure in the previous k months. 
    c_k: the conditional probability that an individual had zero 
         expenditure in the current month given that the individual also
         had zero expenditure in the previous k months. 
    --------------------------------------------------------------------
    INPUTS:
    data      = a (n, k) pandas.DataFrame of healthcare expenditure
    uncondPr  = a boolean
                    if True returns a numpy.array of length 2,
                        which contiins b_k and d_k
                    if false returns a numpy.array of length 2,
                        which contains a_k and c_k

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    months    = integer, the total number of time periods, k
    ak_vec    = a numpy.array, in which each element represents
                the probability of an individual had positive expenditure
                in the current month given that the individual also had
                positive expenditure in the previous k months
    ck_vec    = a numpy.array, in which each element represents
                the probability of an individual had zero expenditure
                in the current month given that the individual also had
                zero expenditure in the previous k months
    data_vals = a numpy.array of the same dimension as data input
    ak0       = the probability of positive expenditure given positive
                expenditure of previous 0 month -- special case
    ck0       = the probability of positive expenditure given positive
                expenditure of previous 0 month -- special case
    pos_count = a 1D numpy.array of length n, in which each element 
                represents the number of occurrences of previous k periods 
                of positive expenditure
    pos_pos   = a 1D numpy.array of length n, in which each element represents
                the number of occurrences of positive expenditures of
                the current month given previous k months of positive 
                expenditures
    zero_count= a 1D numpy.array of length n, in which each element represents
                the number of occurrences of previous k periods of
                zero expenditure
    zero_zero = a 1D numpy.array of length n, in which each element represents
                the number of occurrences of zero expenditures of
                current month given previous k months of zero expenditures
    bk_vec    = a numpy.array, in which each element is the unconditional
                probability of positive health expenditure in the current month
                given previous k months also had positive health expenditure
    dk_vec    = a numpy.array, in which each element is the unconditional
                probability of zero health expenditure in the current month
                given previous k months also had zero health expenditure

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: bk_vec
    --------------------------------------------------------------------
    '''
    months = data.shape[1]
    ak_vec = np.zeros(months)
    ck_vec = np.zeros(months)
    # convert data structure from pd to np to facilitate vector operation
    data_vals = data.values

    for k in range(months):
        # special case when k = 0: compute the Pr of monthly
        # + expenditure given previous 0 months have + expenditure
        if k == 0:
            ak0 = (data_vals > 0).sum((0, 1))/np.prod(data.shape)
            ck0 = (data == 0.0).sum().sum()/np.prod(data.shape)
            ak_vec[k] = ak0
            ck_vec[k] = ck0
        else:
            pos_count = np.zeros(data_vals.shape[0])
            pos_pos = np.zeros(data_vals.shape[0])
            zero_count = np.zeros(data_vals.shape[0])
            zero_zero = np.zeros(data_vals.shape[0])
            for t in range(k, months):
                pos_count += np.prod(data_vals[:,t-k:t] > 0, axis=1)
                pos_pos += np.prod(data_vals[:,t-k:t+1] > 0, axis=1)
                zero_count += np.prod(data_vals[:,t-k:t] == 0, axis=1)
                zero_zero += np.prod(data_vals[:,t-k:t+1] == 0, axis=1)
            ak_vec[k] = pos_pos.sum()/pos_count.sum()
            ck_vec[k] = zero_zero.sum()/zero_count.sum()

    bk_vec = np.cumprod(ak_vec)
    dk_vec = np.cumprod(ck_vec)

    if uncondPr:
        return np.array([bk_vec, dk_vec])
    else:
        return np.array([ak_vec, ck_vec])


'''
---------------------------------------------------------------------------
II. Define erg_dist()
---------------------------------------------------------------------------
'''
# define functions for getting a vector of ergodict distribution
def erg_dist_vec(*params):

    lambda_vec, delta_vec = params
    lambda0, lambda1, lambda2, lambda3, lambda4 = lambda_vec
    delta1, delta2, delta3 = delta_vec
    delta4 = 1 - lambda0 - delta1 - delta2 - delta3

    # get ergodic distribution vector
    # Markov transition matrix A
    A = np.array([ [lambda0, delta1, delta2, delta3, delta4],
                   [1 - lambda1, lambda1, 0, 0, 0],
                   [1 - lambda2, 0, lambda2, 0, 0],
                   [1 - lambda3, 0, 0, lambda3, 0],
                   [1 - lambda4, 0, 0, 0, lambda4] ])

    # a robust way to get the ergodic distribution vector
    A_erg = LA.matrix_power(A, 500)
    erg_dist = A_erg[0, :]

    return erg_dist


'''
---------------------------------------------------------------------------
III. Define model_moments()
---------------------------------------------------------------------------
'''
def model_moments(*params):
    '''
    --------------------------------------------------------------------
    This function computes the model moments, bk and dk

    model assumption: page5
    if an individual leaves a positive expenditure state or the very
    health state, they must spend at least one period in the basic
    health state H0 before returning to another positive expenditure
    state or to the very healthy state
    --------------------------------------------------------------------
    INPUTS:
    params  = a (3,) numpy.array, wherein the elements include
              n, delta_vec and lambda_vec


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    n                 = integer, the number of time periods
    delta1 ~ delta4   = delta parameters
    lambda0 ~ lambda4 = labmda parameters
    A                 = a (5,5) numpy.array, the Markov transition matrix
    erg_dist          = a (5,) numpy.array, the ergodic distribution of
                        health states
    bk_vec            = a numpy.array of probabilities with each element to be
                        the unconditional probability of positive health
                        expenditure in the current month given previous k
                        months also had positive health expenditure
    p_k0              = an numpy array of dimension (n,), which is an array
                        of unconditional probabilites of an individual
                        would be in state H0 at time k
    p_k1              = an numpy array of dimension (n,), which is an array
                        of the unconditional probability of an individual
                        would be in state H1 at time k
    dk_vec            = a numpy array of probabilities with each element to be
                        the unconditional probability of zero health
                        expenditure in the current month given previous k
                        months also had zero health expenditure

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: bk_vec, dk_vec
    --------------------------------------------------------------------
    '''
    lambda_vec, delta_vec, months = params
    lambda0, lambda1, lambda2, lambda3, lambda4 = lambda_vec
    delta1, delta2, delta3 = delta_vec
    delta4 = 1 - lambda0 - delta1 - delta2 - delta3

    # get ergodic distribution vector
    erg_dist = erg_dist_vec(lambda_vec, delta_vec)

    # get model moments
    bk_vec = np.zeros(months)
    dk_vec = np.zeros(months)
    p_k0 = np.zeros(months)
    p_k1 = np.zeros(months)

    for k in range(months):
        # notice model assumption: must get back to the state H0 before
        # going to other state
        bk_vec[k] = lambda2**k * erg_dist[2] + lambda3**k * erg_dist[3] + \
                    lambda4**k * erg_dist[4]
        # p_00 = v0  and p_01 = v1
        if k == 0:
            p_k0[0] = erg_dist[0]
            p_k1[0] = erg_dist[1]
        else:
            p_k0[k] = p_k0[k-1] * lambda0 + p_k1[k-1] * (1 - lambda1)
            p_k1[k] = p_k0[k-1] * delta1 + p_k1[k-1] * lambda1

    # special case when k = 0
    dk_vec[0] = 1 - bk_vec[0]
    # normal case from k = (1, 23)
    dk_vec[1:] = p_k0[:-1] * (lambda0 + delta1) + p_k1[:-1]

    return np.array([bk_vec, dk_vec])


'''
---------------------------------------------------------------------------
IV. Define criterion()
---------------------------------------------------------------------------
'''
def criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params: all parameters in the order of
                lambda0 - lambda4 followed by delta1-delta3 
    args  : all arguments that would require for the optimization algorithm
                - bk_vec_data: data moments bk, which is a vector of 
                               the unconditional probability that the
                               current period and the previous k periods all 
                               had positive expenditures. 
                - dk_vec_data: data moments dk, which is a vector of 
                               the unconditional probability that the
                               current period and the previous k periods all 
                               had zero expenditures. 
                - W          : the weighting matrix
                - months     : the total number of periods 
                - simple     : boolean, =True if errors are simple difference,
                                        =False if errors are percent deviation 
                                         from data moments
 
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        model_moments()
 
    OBJECTS CREATED WITHIN FUNCTION:
    err        = column vector of two moment error functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    lambda_vec, delta_vec = params[:5], params[5:]
    bk_vec_data, dk_vec_data, W, months, simple = args
    bk_vec_mod, dk_vec_mod = model_moments(lambda_vec, delta_vec, months)

    data_moms = np.append(bk_vec_data, dk_vec_data[1:])
    model_moms = np.append(bk_vec_mod, dk_vec_mod[1:])

    if simple == True:
        err = model_moms - data_moms
    else:
        err = (model_moms - data_moms) / data_moms

    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val


'''
---------------------------------------------------------------------------
V. Define spell_prob()
---------------------------------------------------------------------------
'''
def spell_prob(*params):
    lambda_vec, delta_vec, months, censor_type = params
    erg_dist = erg_dist_vec(lambda_vec, delta_vec[:-1])

    if censor_type == 'none':
        censored_none = []
        for i in range(3):
            s = i + 2
            prob = np.zeros(months-2)
            for t in range(months-2):
                n = t + 1
                prob[t] = delta_vec[s-1] * lambda_vec[s]**(n-1) * (1-lambda_vec[s])
            censored_none.append(prob)
        outcome = np.array(censored_none)/np.array(censored_none).sum(axis=0)
    elif censor_type == 'both':
        censored_both = []
        for i in range(3):
            s = i + 2
            prob = np.zeros(months)
            for t in range(months):
                n = t + 1
                prob[t] = erg_dist[s] * lambda_vec[s]**(n-1)
            censored_both.append(prob)
        outcome = np.array(censored_both)/np.array(censored_both).sum(axis=0)
    elif censor_type == 'front':
        censored_front = []
        for i in range(3):
            s = i + 2
            prob = np.zeros(months-1)
            for t in range(months-1):
                n = t + 1
                prob[t] = erg_dist[s] * lambda_vec[s]**(n-1) * (1-lambda_vec[s])
            censored_front.append(prob)
        outcome = np.array(censored_front)/np.array(censored_front).sum(axis=0)
    elif censor_type == 'back':
        censored_back = []
        for i in range(3):
            s = i + 2
            prob = np.zeros(months-1)
            for t in range(months-1):
                n = t + 1
                prob[t] = delta_vec[s-1] * lambda_vec[s]**(n-1)
            censored_back.append(prob)
        outcome = np.array(censored_back)/np.array(censored_back).sum(axis=0)

    return outcome
