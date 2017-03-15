
# coding: utf-8

# In[230]:

import numpy as np
import numpy.linalg as lin
import pandas as pd
import math 
import scipy.optimize as opt
import matplotlib.pyplot as plt


# In[348]:

df = pd.read_stata('/Users/ale/Desktop/MAPSS/Winter/StructuralEstimation/project_merged_1.dta')


# In[280]:

# Data Moments

profits_grby_state = df['Pibar2007'].groupby(df['marketstate2007'])
RD_grby_state = df['RDbar2007'].groupby(df['marketstate2007'])

Pibar = profits_grby_state.mean()
RDbar = RD_grby_state.mean()

mu_0 = profits_grby_state.size()[0]/137
mu_1 = profits_grby_state.size()[1]/137

pi_0_bar = Pibar[0]
pi_1_bar = Pibar[1]

n_0_bar = RDbar[0]
n_1_bar = RDbar[1]


m1 = pi_0_bar/pi_1_bar
m2 = n_0_bar
m3 = n_1_bar 
m4 = pi_1_bar
m5 = mu_0/mu_1 


# In[354]:

def datamoments_GMM(eps,gam,h,E):
    
    delta = 1 - eps
    
    pred_pi_1 = E*(1-(1/gam))
    
    pred_n_0 = - h + math.sqrt((h**2) + 2*delta*pred_pi_1)
    
    pred_n_1 = - (h + pred_n_0) + math.sqrt((h**2) + (pred_n_0**2) + (2*delta*pred_pi_1))
    
    e1 = (m1-eps)/m1 
    
    e2 = (m2 - pred_n_0)/m2
    
    e3 = (m3 - pred_n_1) / m3
     
    e4 = (m4 - pred_pi_1)/m4
    
    
    e5 = ( m5 - (pred_n_1 + h) / (2*pred_n_0) ) / m5
    
    e = np.array([e1, e2, e3, e4, e5]).reshape(5,1)
    
    return e

def criterion_GMM(params,*args):
    eps,gam,h,E = params
    W = args
    err = datamoments_GMM(eps,gam,h,E)
    crit = np.dot(np.dot(err.T, W), err)
    return crit

# Initial parameters and arguments
eps_0 = 0.1
gam_0 = 10
h_0 = 0.9
E_0 = 20000
params_0 = np.array([eps_0,gam_0,h_0,E_0])

# First Step Estimation

W = np.identity(5)
GMM_args = (W)

results_GMM = opt.minimize(criterion_GMM, params_0, args=(GMM_args),
                       method='L-BFGS-B', bounds=((0.001,0.5),(1.001,None),(0.001,0.999),(0.001,None)))


GMM_eps = results_GMM.x[0]
GMM_gam = results_GMM.x[1]
GMM_h = results_GMM.x[2]
GMM_E = results_GMM.x[3]

print("Results First Step:")
print("Criterion Function:",results_GMM.fun[0][0][0])
print("Epsilon:",GMM_eps)
print("Gamma:",GMM_gam)
print("h:",GMM_h)
print("E:",GMM_E)


## Moments from Estimates

def GMM_moments(eps,gam,h,E):
    delta = 1 - eps
    prof_ratio = eps
    pi_1 = E*(1-(1/gam))
    n_0 = - h + math.sqrt((h**2) + 2*delta*pi_1)
    n_1 = - (h + n_0) + math.sqrt((h**2) + (n_0**2) + (2*delta*pi_1))
    relative_share = (n_1 + h)/(2*n_0)
    moments = np.array([prof_ratio, n_0, n_1, pi_1,relative_share]).reshape(5,1)
    growth = 2 * n_0 * (1/(1+(1/relative_share))) + (1/(relative_share+1)) * (h + n_1)
    return moments, growth

init_moments = GMM_moments(eps_0,gam_0,h_0,E_0)[0]
GMM_mom = GMM_moments(GMM_eps,GMM_gam,GMM_h,GMM_E)[0]

## Errors from Estimates
GMM_e1 = (m1-GMM_mom[0])/m1 
GMM_e2 = (m2 - GMM_mom[1])/m2
GMM_e3 = (m3 - GMM_mom[2]) / m3
GMM_e4 = (m4 - GMM_mom[3])/m4
GMM_e5 = (m5 - GMM_mom[4])/m5
GMM_e = np.array([GMM_e1, GMM_e2, GMM_e3, GMM_e4, GMM_e5]).reshape(5,1)

# Obtain the Optimal Weighting Matrix
VCV2 = np.dot(GMM_e, GMM_e.T) / 137
W_hat2 = lin.pinv(VCV2) 
GMM_args_2S = (W_hat2)

# New Initial Guesses
eps_0_2 = GMM_eps
gam_0_2 = GMM_gam
h_0_2 = GMM_h
E_0_2 = GMM_E

params_0_2 = np.array([eps_0_2,gam_0_2,h_0_2,E_0_2])


results_GMM_2S = opt.minimize(criterion_GMM, params_0_2, args=(GMM_args_2S),
                       method='L-BFGS-B', bounds=((0.001,0.5),(1.001,None),(0.001,0.999),(0.001,None)))

GMM_eps_2 = results_GMM_2S.x[0]
GMM_gam_2 = results_GMM_2S.x[1]
GMM_h_2 = results_GMM_2S.x[2]
GMM_E_2 = results_GMM_2S.x[3]

print("")
print("Results Second Step:")
print("Criterion Function:",results_GMM_2S.fun[0][0][0])
print("Epsilon:",GMM_eps_2)
print("Gamma:",GMM_gam_2)
print("h:",GMM_h_2)
print("E:",GMM_E_2)

GMM2S_mom = GMM_moments(GMM_eps_2,GMM_gam_2,GMM_h_2,GMM_E_2)[0]


# In[351]:

#  Moments



print("Data Moments","Initial Parameter Moments","GMM First Step Moments","GMM Second Step Moments")
print(m1,init_moments[0],GMM_mom[0],GMM2S_mom[0][0])
print(m2,init_moments[1],GMM_mom[1],GMM2S_mom[1][0])
print(m3,init_moments[2],GMM_mom[2],GMM2S_mom[2][0])
print(m4,init_moments[3],GMM_mom[3],GMM2S_mom[3][0])
print(m5,init_moments[4],GMM_mom[4],GMM2S_mom[4][0])


# In[362]:




growth_base = GMM_moments(GMM_eps_2,GMM_gam_2,GMM_h_2,GMM_E_2)[1]

def growth_eps(eps):
    g = GMM_moments(eps,GMM_gam_2,GMM_h_2,GMM_E_2)[1]
    return g

growth_maxeps = growth_eps(0.5)
growth_loweps = growth_eps(0)

print("Changes in epsilon")
print("eps=e_GMM_2:",growth_base,"eps=0.5:",growth_maxeps,"eps=0:",growth_loweps)

def growth_h(h):
    g = GMM_moments(GMM_eps_2,GMM_gam_2,h,GMM_E_2)[1]
    return g

growth_maxh = growth_h(1)
growth_lowh = growth_h(0)

print("Changes in h")
print("h=h_GMM_2:",growth_base,"h=1:",growth_maxh,"h=0:",growth_lowh)


# In[363]:

y1 = growth_eps(0.05)
y2 = growth_eps(0.1)
y3 = growth_eps(0.15)
y4 = growth_eps(0.2)
y5 = growth_eps(0.25)
y6 = growth_eps(0.3)
y7 = growth_eps(0.35)
y8 = growth_eps(0.4)
y9 = growth_eps(0.45)
y10 = growth_eps(0.5)

eps = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])
geps = np.array([y1,y2,y3,y4,y5,y6,y7,y8,y9,y10])

plt.plot(eps,geps)
plt.xlabel('Degree of Collusion (epsilon)')

plt.ylabel('Growth')
plt.show()


# In[364]:

g0 = growth_h(0)
g1 = growth_h(0.1)
g2 = growth_h(0.2)
g3 = growth_h(0.3)
g4 = growth_h(0.4)
g5 = growth_h(0.5)
g6 = growth_h(0.6)
g7 = growth_h(0.7)
g8 = growth_h(0.8)
g9 = growth_h(0.9)
g10 = growth_h(1)

h = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
gh = np.array([g0,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10])

plt.plot(h,gh)
plt.xlabel('Catch-up rate (h)')
plt.ylabel('Growth')
plt.show()

