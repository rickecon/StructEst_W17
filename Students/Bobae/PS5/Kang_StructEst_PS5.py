'''
--------------------------------------------------------------------
This is Bobae's Python script for MACS 40200 PS5
--------------------------------------------------------------------
--------------------------------------------------------------------
Preparatory steps
--------------------------------------------------------------------
'''
# Import packages and load the data
import numpy as np
import pandas as pd
import scipy.stats as sts
import scipy.optimize as opt

# import data
data = pd.read_csv('MacroSeries.txt', header = None)

# define necessary functions
def moments_gmm(data, params):
    c, k, w, r = np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3])
    ct, ct1 = c[:-1], c[1:]
    wt, wt1 = w[:-1], w[1:]
    kt, kt1 = k[:-1], k[1:]
    rt, rt1 = r[:-1], r[1:]

    alpha, beta, rho, mu = params

    zt = np.log(rt) - np.log(alpha) - (np.log(kt) * (alpha - 1))
    zt1 = np.log(rt1) - np.log(alpha) - (np.log(kt1) * (alpha - 1))

    mom1 = zt1 - (rho * zt) - ((1 - rho) * mu)
    mom2 = mom1 * zt
    mom3 = (beta * alpha * np.exp(zt1) * (kt1**(alpha - 1)) * (ct / ct1)) - 1
    mom4 = mom3 * wt

    return mom1.mean(), mom2.mean(), mom3.mean(), mom4.mean()

def criterion_gmm(params, *args):
    data, W = args
    err = np.array(moments_gmm(data, params))
    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val

def data_moments_smm(data):
    ct, wt, kt, rt = np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3])
    kt1 = wt + rt*kt - ct

    mom1_dat = ct.mean()
    mom2_dat = kt.mean()
    mom3_dat = ct.var()
    mom4_dat = kt.var()
    mom5_dat = sts.pearsonr(ct, kt)[0]
    mom6_dat = sts.pearsonr(kt, kt1)[0]

    return mom1_dat, mom2_dat, mom3_dat, mom4_dat, mom5_dat, mom6_dat

def model_moments_smm(eps, params):
    alpha, beta, rho, mu, sigma = params

    zt_sim = np.zeros([T,S])
    for i in list(range(0,T)):
        if i == 0:
            zt_sim[i,:] = mu
        else:
            zt_sim[i,:] = (rho * zt_sim[i-1,:]) + ((1 - rho) * mu) + eps[i,:]
    kt_sim = np.zeros([T,S])
    for i in list(range(0,T)):
        if i == 0:
            kt_sim[i,:] = k.mean()
        else:
            kt_sim[i,:] = alpha * beta * np.exp(zt_sim[i,:]) * (kt_sim[i-1,:]**alpha)

    kt1_sim = alpha * beta * np.exp(zt_sim) * (kt_sim**alpha)
    wt_sim = (1 - alpha) * np.exp(zt_sim) * (kt_sim**alpha)
    rt_sim = alpha * np.exp(zt_sim) * (kt_sim**(alpha - 1))
    ct_sim = wt_sim + (rt_sim * kt_sim) - kt1_sim

    mom5_vec = np.zeros([S, 1])
    for i in list(range(0,S)):
        mom5_vec[i,] = sts.pearsonr(ct_sim[:,i], kt_sim[:,i])[0]
    mom6_vec = np.zeros([S, 1])
    for i in list(range(0,S)):
        mom6_vec[i,] = sts.pearsonr(kt_sim[:,i], kt1_sim[:,i])[0]

    mom1_mod = np.mean(ct_sim.mean(axis=0))
    mom2_mod = np.mean(kt_sim.mean(axis=0))
    mom3_mod = np.mean(ct_sim.var(axis=0))
    mom4_mod = np.mean(kt_sim.var(axis=0))
    mom5_mod = np.mean(mom5_vec)
    mom6_mod = np.mean(mom6_vec)

    return mom1_mod, mom2_mod, mom3_mod, mom4_mod, mom5_mod, mom6_mod

def criterion_smm(params, *args):
    data, eps, W = args
    moms_dat = data_moments_smm(data)
    moms_mod = model_moments_smm(eps, params)
    err = (np.array(moms_mod) - np.array(moms_dat))/np.array(moms_dat)
    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val

'''
--------------------------------------------------------------------
1. Estimating the Brock and Mirman (1972) model by GMM
--------------------------------------------------------------------
'''
# setup for optimization
alpha_0, beta_0, rho_0, mu_0 = .9, .9, .8, .9 # converges with criterion value = 0.0010927458976064087
params_gmm_0 = np.array([alpha_0, beta_0, rho_0, mu_0])
W_gmm = np.eye(4) # weight matrix = I
args_gmm_0 = (data, W_gmm)
bounds_gmm = ((1e-16, 1-1e-16), (1e-16, 1-1e-16), (-1+1e-16, 1-1e-16), (1e-16, None))

# optimization
results_1 = opt.minimize(criterion_gmm, params_gmm_0, args=(args_gmm_0),
                            method='L-BFGS-B', # 'TNC', 'L-BFGS-B', 'SLSQP'
                            bounds=bounds_gmm
                            )
# results
alpha_1, beta_1, rho_1, mu_1 = results_1.x

def problem1():
    # report the results
    print('Answers to part 1')
    print('GMM estimation for alpha: {}'.format(alpha_1),
        '\nGMM estimation for beta: {}'.format(beta_1),
        '\nGMM estimation for rho: {}'.format(rho_1),
        '\nGMM estimation for mu: {}'.format(mu_1))
    print('Criterion function value: {}'.format(results_1.fun))
    print('Optimization result 1: \n{}'.format(results_1))

'''
--------------------------------------------------------------------
2. Estimating the Brock and Mirman (1972) model by SMM
--------------------------------------------------------------------
'''
# setup for optimization
T = 100 # number of time points
S = 1000 # number of simulations
k = np.array(data[2]) # to get the k0 for simulating kt values
sig_0 = 0.05 # converged with GMM estimates
eps = sts.norm.rvs(loc=0, scale=sig_0, size = (T,S)) # simulated errors
params_smm_0 = np.array([alpha_1, beta_1, rho_1, mu_1, sig_0]) # using GMM estimates
W_smm = np.eye(6) # weight matrix = I
args_smm_0 = (data, eps, W_smm)
bounds_smm = ((.01, .99),(.01, .99),(-.99, .99),(-.5, 1.),(.001, 1.))

# optimization
results_2 = opt.minimize(criterion_smm, params_smm_0, args=(args_smm_0),
                            method='L-BFGS-B', # 'TNC', 'L-BFGS-B', 'SLSQP'
                            bounds=bounds_smm#, options = {'eps':1.}
                            )
# results
alpha_2, beta_2, rho_2, mu_2, sig_2 = results_2.x
params_2 = np.array([alpha_2, beta_2, rho_2, mu_2, sig_2])
mom_diff = np.array(model_moments_smm(eps, params_2)) - np.array(data_moments_smm(data))
mom_diff_percent = mom_diff/np.array(data_moments_smm(data)) * 100

def problem2():
    # report the results
    print('\nAnswers to part 2')
    print('SMM estimation for alpha: {}'.format(alpha_2),
        '\nSMM estimation for beta: {}'.format(beta_2),
        '\nSMM estimation for rho: {}'.format(rho_2),
        '\nSMM estimation for mu: {}'.format(mu_2),
        '\nSMM estimation for sigma: {}'.format(sig_2))
    print('Moment differences: {}'.format(mom_diff))
    print('Moment percent differences: {}'.format(mom_diff_percent))
    print('Criterion function value: {}'.format(results_2.fun))
    print('Optimization result 2: \n{}'.format(results_2))

'''
--------------------------------------------------------------------
Answers.
--------------------------------------------------------------------
'''
def main():
    problem1()
    problem2()

if __name__ == '__main__':
	main()
