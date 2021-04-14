import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from TA import colormaps as cmaps

def plotInterp(data):
    x = np.linspace(0,1,len(data['delays']))
    t = data['delays']
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[cmaps.parula(i) for i in np.linspace(0,1,len(t))])
    mpl.rcParams.update({'font.size': 7,'figure.figsize' : [1.569,1.569]}) # Font size
    wL = data['x']
    wLnew = np.linspace(min(wL), max(wL), 1000)
    maxval = max(np.average(data['dA'], axis=2).flatten())
    for dA in np.average(data['dA'], axis=2).T/maxval:
        interp_dA = interp1d(wL, dA, kind='cubic')
        plt.plot(wLnew, interp_dA(wLnew),zorder=1, lw=0.2)
        plt.scatter(wL,dA,color=[0.2,0.2,0.2],s=0.1,zorder=2,lw=0)
   # plt.tight_layout()

def plotFit( data, KS, model, pixels ):
    mpl.rcParams.update({'font.size': 7,'figure.figsize' : [1.569,1.569]}) # Font size
    dA = np.average(data['dA'], axis=2)
    maxval = max(dA.flatten())
    dA = dA/maxval
    


 
def plotSAS( filenames ):
    mpl.rcParams.update({'font.size': 18,"font.weight":"normal",
                         'figure.figsize' : [6,10]}) # Font size, and make bold
    for filename in filenames:
        spec = np.loadtxt(filename)
        wL = spec[0]
        dA = spec[1]
        dA = dA/max(dA)
        interp_dA = interp1d(wL, dA, kind='cubic')
        wLnew = np.linspace(min(wL), max(wL), 1000)
        plt.plot(wLnew, interp_dA(wLnew),zorder=1, lw=2)
        plt.scatter(wL,dA,color=[0.25,0.25,0.25],s=5,zorder=2)
    #plt.tight_layout()
