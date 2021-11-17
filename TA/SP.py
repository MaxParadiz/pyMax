import numpy as np
from scipy.special import erf

def gaussexp(t,G):
 nK = int((len(G)-2)/2)
 c = G[0]
 s = G[1]
 nt = t-c
 A = G[2:2+nK]
 K = abs(G[2+nK::])
 return A@np.array([0.5*np.exp(-k*(nt-0.5*s**2*k))*(1+erf((nt-s**2*k)/(2**0.5*s))) for k in K])

def resid(G,func,t,y):
 return func(t,G)-y

