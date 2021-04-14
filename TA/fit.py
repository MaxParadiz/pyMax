import numpy as np
from scipy.optimize import least_squares
from TA import KModels
 


def residuals_floppy(G,data, kr,phi,nComponents,scan=False, average=False):
     time = data['delays']
     npixels = data['npixels']
     W = G[:npixels*nComponents].reshape(npixels,nComponents)
     K = G[npixels*nComponents:]
     k1 = K[0]
     knr = K[1]
     ## Special k-values here
     k2 = (knr*(k1+kr-kr/phi))/(kr/phi-kr)
     ###########################
     y = np.dot(W,np.vstack([np.squeeze(KModels.FloppyRotor(time,k1,kr,knr,k2,1.0,0.0))])) 
     if scan != False:
         return ((y[:,:] - data['dA'][:,:,scan])).flatten()
     elif average == False:
         return ((y[:,:,None] - data['dA'][:,:,:])).flatten()
     else:
         return ((y[:,:] - np.average(data['dA'],axis=2))).flatten()


def residuals_floppy_baseshift(G,data, kr,phi,nComponents,scan=False, average=False):
     time = data['delays']
     npixels = data['npixels']
     W = G[:npixels*nComponents].reshape(npixels,nComponents)
     K = G[npixels*nComponents:]
     k1 = K[0]
     knr = K[1]
     ## Special k-values here
     k2 = (knr*(k1+kr-kr/phi))/(kr/phi-kr)
     ###########################
     y = np.dot(W,np.vstack([np.squeeze(KModels.FloppyRotor(time,k1,kr,knr,k2,1.0,0.0)), np.ones(len(time))])) 
     if scan != False:
         return ((y[:,:] - data['dA'][:,:,scan])).flatten()
     elif average == False:
         return ((y[:,:,None] - data['dA'][:,:,:])).flatten()
     else:
         return ((y[:,:] - np.average(data['dA'],axis=2))).flatten()



def residuals_sequential(G,data,kr,phi,nComponents,scan=None, average=False):
     time = data['delays']
     npixels = data['npixels']
     W = G[:npixels*nComponents].reshape(npixels,nComponents)
     K = G[npixels*nComponents:]
     k1 = K[0]
     knr = K[1]
     y = np.dot(W,np.vstack([np.squeeze(KModels.Sequential(time,k1,kr,knr,1.0,0.0))])) 
     if scan != None:
         return ((y[:,:] - data['dA'][:,:,scan])).flatten()
     elif average == False:
         return ((y[:,:,None] - data['dA'][:,:,:])).flatten()
     else:
         return ((y[:,:] - np.average(data['dA'],axis=2))).flatten()

def residuals_sequential_baseshift(G,data,kr,phi,nComponents,scan=None, average=False):
     time = data['delays']
     npixels = data['npixels']
     W = G[:npixels*nComponents].reshape(npixels,nComponents)
     K = G[npixels*nComponents:]
     k1 = K[0]
     knr = K[1]
     y = np.dot(W,np.vstack([np.squeeze(KModels.Sequential(time,k1,kr,knr,1.0,0.0)),np.ones(len(time))])) 
     if scan != None:
         return ((y[:,:] - data['dA'][:,:,scan])).flatten()
     elif average == False:
         return ((y[:,:,None] - data['dA'][:,:,:])).flatten()
     else:
         return ((y[:,:] - np.average(data['dA'],axis=2))).flatten()




def fit( model, data, GSAS=False, K0 = [0.1,0.02], scan=None, average=False,kr = 0.00014, nComponents =2, phi=1):
    residuals = {'FloppyRotor': residuals_floppy,'Sequential': residuals_sequential,
                 'SequentialB': residuals_sequential_baseshift,
                 'FloppyRotorB': residuals_floppy_baseshift}
    npix = data['npixels']
    if GSAS:
        GSAS = np.loadtxt(GSAS)
    else:
        GSAS = 0.8*np.ones(nComponents*npix)                 # Initial guess for the spectrum of M
        G0 = np.concatenate([GSAS,K0]) 
    res_lsq = least_squares(residuals[model], G0, args=(data,kr, phi,nComponents, scan, average))
    return procFit(data, res_lsq, nComponents=nComponents)



def fitError( data, res_lsq):
    return 0 ### in progress 
#    _, s, VT = np.linalg.svd(res_lsq.jac,full_matrices=False)
#    threshold = np.finfo(float).eps * max(res_lsq.jac.shape)*s[0]
#    s = s[s>threshold]
#    VT = VT[:s.size]
#    pcov = np.dot(VT.T / s**2, VT)
#    s_sq = np.sum((res_lsq.fun**2))/(len(res_lsq.fun.flatten())-len(res_lsq.x))
#    pcov = pcov * s_sq
#    err = np.sqrt(np.diag(pcov))
#    data['K_err'] = err[-2:]
#    return data





def procFit( data, res_lsq, nComponents=2):
    npixels = data['npixels']
    time = data['delays']
    G = res_lsq.x
    W = G[:npixels*nComponents].reshape(npixels,nComponents)
    K = G[npixels*nComponents:]
    data['SAS'] = {}
    for c in range(nComponents):
        data['SAS'][c] = W[:,c]
    data['K'] = G[npixels*nComponents:]
    data['result'] = res_lsq
    return data
