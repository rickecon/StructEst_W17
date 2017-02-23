
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy.integrate as integrate
import math
import scipy.optimize as opt
import numpy.linalg as lin


# In[2]:

dt = np.loadtxt('/Users/ale/Desktop/MAPSS/Winter/StructuralEstimation/PSet_5/MacroSeries.txt',delimiter=',')


# In[3]:

def TFP_GMM(alpha,r,k):
    z = np.log(r) - np.log(alpha) + (1-alpha)*np.log(k)
    z_ante = z[0:99]
    z_post = z[1:100]
    return z_ante,z_post

def datamoments_GMM(alpha,beta,rho,mu,x):
    r_t = x[:,3].reshape((100,1))
    k_t = x[:,1].reshape((100,1))
    z_t,z_t1 = TFP_GMM(alpha,r_t,k_t)
    k_t1 = x[:,1][1:100]
    c_t1 = x[:,0][1:100]
    c_t = x[:,0][0:99]
    w_t = x[:,2][0:99]
    m1 = np.mean(z_t1 - rho*z_t - (1-rho)*mu) 
    m2 = np.mean((z_t1 - rho*z_t - (1-rho)*mu)*z_t)
    m3 = np.mean(beta*alpha*np.exp(z_t1)*np.power(k_t1,alpha-1)*(c_t/c_t1)-1)
    m4 = np.mean((beta*alpha*np.exp(z_t1)*np.power(k_t1,alpha-1)*(c_t/c_t1)-1)*w_t)
    m = np.array([m1, m2, m3, m4]).reshape(4,1)
    return m

def criterion_GMM(params,*args):
    alpha,beta,rho,mu = params
    xvals, W = args
    eps = datamoments_GMM(alpha,beta,rho,mu,xvals)
    crit = np.dot(np.dot(eps.T, W), eps)
    return crit


# In[4]:

# Estimating the initial guesses 
alpha_0 = 0.3
z_0,z_1 = TFP_GMM(alpha_0,dt[:,3].reshape((100,1)),dt[:,1].reshape((100,1)))
rho_0, intercept, r_value, p_value, std_err = sts.linregress(z_0.reshape(99,),z_1.reshape(99,))
mu_0 = intercept/(1-rho_0)
beta_0 = 0.90
params_0 = np.array([alpha_0,beta_0,rho_0,mu_0])


# In[22]:

#Optimization
W = np.identity(4)
GMM_args = (dt, W)

results_GMM = opt.minimize(criterion_GMM, params_0, args=(GMM_args),
                       method='TNC', bounds=((0.001,1), (0.001,1), (-1,1),(0.001,None)))

print("GMM Results")
print(results_GMM)


# In[21]:

# SMM Estimation

k1 = np.mean(dt[:,1])

def simulations(S,T,alpha,beta,rho,mu,sigma):
    res = np.ones((S,T,6))
    # Create epsilon 
    res[:,:,0] = np.random.normal(0,sigma,(S,T))
    res[:,0,1] = mu
    res[:,0,2] = k1
    for t in range(1,T):
        # Create z
        res[:,t,1] = rho * res[:,t-1,1] + (1-rho) * mu + res[:,t,0]
        # Create k
        res[:,t,2] = alpha * beta * np.exp(res[:,t-1,1]) * np.power(res[:,t-1,2],alpha)
    # Create w
    res[:,:,3]= (1-alpha) * np.exp(res[:,:,1]) * np.power(res[:,:,2],alpha)
    # Create r
    res[:,:,4]= alpha * np.exp(res[:,:,1]) * np.power(res[:,:,2],alpha-1)
    # Create c
    k_pre = res[:,:,2][:,range(0,T-1)]
    k_pos = res[:,:,2][:,range(1,T)]
    w_pre = res[:,:,3][:,range(0,T-1)]
    r_pre = res[:,:,4][:,range(0,T-1)]
    c = w_pre + r_pre * k_pre + k_pos
    # Assume in the last period the household
    # does not save thus c = w + r*k
    c_app = res[:,T-1,3] + res[:,T-1,4] * res[:,T-1,2]
    c = np.append(c,c_app.reshape(S,1),1)
    res[:,:,5] = c
    return res

def moment_SMM(S,T,alpha,beta,rho,mu,sigma):
    sim = simulations(S,T,alpha,beta,rho,mu,sigma)
    
    # Mean of c and k
    mn_c = np.ndarray.mean(sim[:,:,5],1)
    
    mn_k = np.ndarray.mean(sim[:,:,2],1)
    
    # Variances of c and k
    va_c = np.ndarray.var(sim[:,:,5],1)
    va_k = np.ndarray.var(sim[:,:,2],1)
    
    # Correlation of c and k
    ck_mat = np.corrcoef(sim[:,:,5],sim[:,:,2])
    ck_cor = ck_mat[0:S,S:(2*S)].diagonal()
    
    # Correlation of k_t and k_t+1
    k_t = sim[:,:,2][:,0:T-1]
    k_t1 = sim[:,:,2][:,1:T]
    kk_mat = np.corrcoef(k_t,k_t1)
    kk_cor = kk_mat[0:S,S:(2*S)].diagonal()
    
    m_1 = np.mean(mn_c)
    m_2 = np.mean(mn_k)
    m_3 = np.mean(va_c)
    m_4 = np.mean(va_k)
    m_5 = np.mean(ck_cor)
    m_6 = np.mean(kk_cor)
    
    m = np.array([m_1,m_2,m_3,m_4,m_5,m_6])
    
    return m

def error_SMM(x,S,T,alpha,beta,rho,mu,sigma):
    m_1_dat = np.mean(x[:,0])
    m_2_dat = np.mean(x[:,1])
    m_3_dat = np.var(x[:,0])
    m_4_dat = np.var(x[:,1])
    m_5_dat = np.corrcoef(x[:,0],x[:,1])[1,0]
    m_6_dat = np.corrcoef(x[:,1][1:100],x[:,1][0:99])[1,0]

    m_dat = np.array([m_1_dat, m_2_dat, m_3_dat, m_4_dat, m_5_dat, m_6_dat])
    m_mod = moment_SMM(S,T,alpha,beta,rho,mu,sigma)
    e = (m_mod - m_dat)/m_dat
    return e
    
    
def criterion_SMM(params,*args):
    alpha,beta,rho,mu,sigma = params
    xvals,S,T, W = args
    eps = error_SMM(xvals,S,T,alpha,beta,rho,mu,sigma)
    crit = np.dot(np.dot(eps.T, W), eps)
    return crit




#Optimization
W_SMM = np.identity(6)
SMM_args = (dt,1000,100, W_SMM)
params_0_SMM = np.array([alpha_0,beta_0,rho_0,mu_0,0.5])

results_SMM = opt.minimize(criterion_SMM, params_0_SMM, args=(SMM_args),
                       method='TNC', bounds=((0.001,0.99), (0.001,0.99), (-0.99,0.99),(-0.5,1),(0.001,1))
                          ,options={'eps': 0.1})

print("SMM Results")
print(results_SMM)


# In[ ]:



