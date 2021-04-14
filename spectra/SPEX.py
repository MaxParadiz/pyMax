import numpy as np
import os

def load_dat(fname):
 data = np.loadtxt(fname,skiprows=2)
 wavelength = data[:,0]
 emission = data[:,1]
 return wavelength, emission




def getQY(A_std, A_sample, phi_std, n_std, n_sample, Em_std, Em_sample):
 F_std = np.sum(Em_std)
 F_sample = np.sum(Em_sample)
 f_std = 1-10**(-A_std)
 f_sample = 1-10**(-A_sample)
 phi_sample = phi_std * F_sample/F_std * f_std/f_sample * n_sample**2 / n_std**2
 return phi_sample



def load_folder(path = "."):
 em_dict = {}
 for fname in [i for i in os.listdir(path) if ".dat" in i]:
  wavelength,emission = load_dat("%s/%s" %(path,fname))
  em_dict[fname[:-4]] = {'wL' : wavelength, 'em': emission}
 return em_dict


