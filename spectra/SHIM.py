import numpy as np


def load_abs(fname):
 absorption_dict = {}
 header = open(fname,'r').readline()
 header = header.replace(' - ...','').split()[2:]
 data = np.loadtxt(fname,skiprows=1)
 wL = data[:,0]
 for i in range(len(header)):
  absorption_dict[header[i]] = {'wL':wL, 'abs': data[:,i+1]}
 return absorption_dict
