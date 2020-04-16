import numpy as np


def remove_background( data , delays=1):
    data['dA'] = data['dA']- np.average(data['dA'][:,0:delays,:], axis=1)[:,None,:]
    return data


def join_spectra( data_sets ):
    return 0


def cutData( data, xmin, xmax, tmin, tmax):
    xmask = (data['x'] >= xmin) * (data['x'] <= xmax)
    tmask = (data['delays'] >= tmin) * (data['delays'] <= tmax)
    data['x'] = data['x'][xmask]
    data['delays'] = data['delays'][tmask]
    data['dA'] = data['dA'][xmask][:,tmask]
    data['dA_err'] = data['dA_err'][xmask][:,tmask]
    data['npixels'] = len(data['x'])
    return data


def proc( data ):
    data = remove_background(data, 3)
    data = cutData(data)
    return data
