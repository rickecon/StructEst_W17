
import pandas as pd
import numpy as np
import scipy.stats as sts
import scipy.optimize as opt

#dt = pd.read_table('MacroSeries.txt',sep=',',header=None,names=['c','k','w','r'])
dt = np.loadtxt('MacroSeries.txt',delimiter=',')


a_0 = 0.3 # Assume the capital share is 30%


# Defining initial parameters for part (a)


z_0_1 = np.log(dt[:,2].reshape((100,1))) - np.log(1-a_0) - a_0*np.log(dt[:,1].reshape((100,1)))


# The initial guess for rho comes from 
# regressing z on its value in the previous period

z_t_1 = z_0_1[1:100]
z_t1_1 = z_0_1[0:99]
rho_0_1 = sts.mstats.linregress(z_t1_1,z_t_1)[0] 

sig_0_1 = np.std(z_0_1)
mu_0_1 = z_0_1[0]

params_0_1 = [a_0,rho_0_1,mu_0_1,sig_0_1]


# Defining initial parameters for part (b)

z_0_2 = np.log(dt[:,3].reshape((100,1))) - (a_0-1)*np.log(dt[:,1].reshape((100,1))) - np.log(a_0) 
z_t_2 = z_0_2[1:100]
z_t1_2 = z_0_2[0:99]

rho_0_2 = sts.mstats.linregress(z_t1_2,z_t_2)[0]

sig_0_2 = np.std(z_0_2)
mu_0_2 = z_0_2[0]

params_0_2 = [a_0,rho_0_2,mu_0_2,sig_0_2]

# Optimization

def TFPnormpdf(wge,rnt,cpt,alpha,rho,mu,sigma,method):
    '''
    The Following function calculates the Normal pdf
    of tfp for a given set of parameters. It starts by
    calculating tfp z from wages (wge), capital (cpt) and
    alpha. It then calculates the normal pdf where the loc
    parameter is a function of the z, rho and mu. The scale 
    is sigma. The method input determines if we calculate 
    TFP using w and k or r and k.
    '''
    if method == 1:
        z = np.log(wge) - np.log(1-alpha) - alpha*np.log(cpt)
    else:
        z = np.log(rnt) - (alpha-1)*np.log(cpt) - np.log(alpha) 
    z_vect = z[1:100]
    mu_n_vect = rho*z[0:99] + (1-rho)*mu
    pdf_vect = sts.norm.pdf(z_vect,loc = mu_n_vect,scale = sigma)
    return pdf_vect

def logliknorm(wge,rnt,cpt,alpha,rho,mu,sigma,method):
    pdf_vect = TFPnormpdf(wge,rnt,cpt,alpha,rho,mu,sigma,method)
    ln_pdf_vect = np.log(pdf_vect)
    likelihood = ln_pdf_vect.sum()
    return likelihood

def crit(parameters,arguments):
    alpha,rho,mu,sigma = parameters
    wge,rnt,cpt,method = arguments
    lklh = logliknorm(wge,rnt,cpt,alpha,rho,mu,sigma,method)
    return -lklh

wgerntcap_1 = [dt[:,2].reshape((100,1)),dt[:,3].reshape((100,1)),dt[:,1].reshape((100,1)),1]
wgerntcap_2 = [dt[:,2].reshape((100,1)),dt[:,3].reshape((100,1)),dt[:,1].reshape((100,1)),2]

bnds = ((0,0.9999),(-1,1),(0, None),(0.01, None))
results_1 = opt.minimize(crit,params_0_1,args= wgerntcap_1,
                       method='SLSQP',bounds=bnds)
results_2 = opt.minimize(crit,params_0_2,args= wgerntcap_2,
                       method='SLSQP',bounds=bnds)

alpha_1 , rho_1, mu_1, sigma_1 = results_1.x
alpha_2 , rho_2, mu_2, sigma_2 = results_2.x

print('Results for part (a)')
print('alpha_MLE=',alpha_1,'rho_MLE=', rho_1,'mu_MLE=',mu_1,'sigma_MLE=', sigma_1)

print('Results for part (b)')
print('alpha_MLE=',alpha_2,'rho_MLE=', rho_2,'mu_MLE=',mu_2,'sigma_MLE=', sigma_2)


results_1.hess_inv
results_1.hess_inv


# Calculate P(r_t>1)

sts.norm.cdf(9.3,loc=9.846,scale=0.0081)


