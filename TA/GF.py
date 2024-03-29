import numpy as np
from sympy import *
from scipy.optimize import least_squares
from scipy.special import erf


def Residuals(G,t,dA):
 wL_length = dA.shape[0] 
 NumKs = len(G)%wL_length
 K = G[0:NumKs]
 W = G[NumKs:].reshape(wL_length,-1)
 C = np.array([np.exp(-k*t) for k in K])
 Y = W@C
 return (Y-dA).flatten()


def fit(t,dA,G0 = 0):
 if G0 == 0:
  G0 = np.concatenate([[1/10],dA[:,1]])
 res_lsq = least_squares(Residuals, G0, args=(t,dA))
 return res_lsq



#def Residuals_conv(G,t,dA):
# wL_length = dA.shape[1]
# NumKs = len(G)%wL_length-2
# K = G[0:NumKs]
# c = G[NumKs]
# s = G[NumKs+1]
# W = G[NumKs+2:].reshape(len(K),-1)
# nt = t-c
# C = np.array([0.5*np.exp(-k*(nt-0.5*s**2*k))*(1+erf((nt-s**2*k)/(2**0.5*s))) for k in K]).T
# Y = C@W
# return (Y-dA).flatten()



def Residuals_conv(G,t,dA):
 wL_length = dA.shape[1]
 NumKs = len(G)%wL_length-2
 K = G[0:NumKs]
 c = G[NumKs]
 s = G[NumKs+1]
 W = G[NumKs+2:].reshape(len(K),-1)
 nt = t-c
 C = np.array([0.5*np.exp(-k*(nt-0.5*s**2*k))*(1+erf((nt-s**2*k)/(2**0.5*s))) for k in K]).T
 Y = C@W
 return (Y-dA).flatten()



def fit_conv(t,dA,G0 = 0):
 if G0.any() == 0:
  G0 = np.concatenate([[1/10,0,0.005],dA[:,1]])
 res_lsq = least_squares(Residuals_conv, G0, args=(t,dA))
 return res_lsq


def load_NS(filename):
 M = np.loadtxt(filename,delimiter=',')
 wL = M[0,1:]
 t = M[1:,0]
 Fl = M[1:,1:]
 return t, wL, Fl


def Model(G,t,dA):
 wL_length = dA.shape[0]
 NumKs = len(G)%wL_length-2
 K = G[0:NumKs]
 c = G[NumKs]
 s = G[NumKs+1]
 W = G[NumKs+2:].reshape(wL_length,-1)
 nt = t-c
 C = np.array([0.5*np.exp(-k*(nt-0.5*s**2*k))*(1+erf((nt-s**2*k)/(2**0.5*s))) for k in K])
 Y = W@C
 return dA
