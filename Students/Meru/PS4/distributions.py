import numpy as np
import scipy.special as spc

def norm_pdf(x,mu,sigma):
	return (1/(sigma*(2*np.pi)**(1/2)))*np.exp(-((x-mu)**2)/(2*sigma**2))

def gamma_pdf(x,alpha,beta):
	return (x**(alpha-1)*np.exp(-x/beta))/(beta**alpha * spc.gamma(alpha))

def ggamma_pdf(x,alpha,beta,m):
	return (m*x**(alpha-1)*np.exp(-(x/beta)**m))/(beta**alpha * spc.gamma(alpha/m))

def gbeta2_pdf(x, a, b, p, q):
    return (a*x**(a*p-1))/(b**(a*p)*spc.beta(p,q)*(1+(x/b)**a)**(p+q))

def log_norm_pdf(x, mu, sigma):
    return (1/(x* sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.log(x) - mu)**2 / (2 * sigma**2)))