import numpy as np


def remove_background( data , delays=1):
    data['dA'] = data['dA']- np.average(data['dA'][:,0:delays,:], axis=1)[:,None,:]
    return data


def join_spectra( data_sets ):
    if len( data_sets ) == 1:
        return 0
    linked = {}
    linked['data_sets'] = data_sets
    linked['x'] = np.concatenate([data['x'] for data in data_sets])
    linked['delays'] = np.concatenate([data['delays'] for data in data_sets])
    linked['dA'] = np.concatenate([np.average(data['dA'], axis=2) for data in data_sets])[linked['x'].argsort()]
    linked['dA_err'] = np.concatenate([np.std(data['dA'], axis=2) for data in data_sets])[linked['x'].argsort()]
    linked['npixels'] = len(linked['x'])
    linked['x'] = np.sort(linked['x'])
    return linked

def removeScan( data, scans):
    if scans == None:
        return data
    data['nscans'] = data['nscans'] - len(scans)
    data['dA'] = np.delete(data['dA'], scans,2)
    data['dA_err'] = np.delete(data['dA_err'], scans,2)
    return data

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
