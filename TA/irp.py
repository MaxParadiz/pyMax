# Max's IRParsing script
# this is used to load data from the legend

import numpy as np



# This function will be given:
# - center pixel number
# - wavelength at center
# - number of pixels
# - type of grating
# The grating is for the
# Oriel MS260I spectrogrph
# website: https://www.newport.com/f/diffraction-gratings-for-cs260-monochromators-and-ms260i-spectrographs
# useful manual: http://www.amstechnologies.com/fileadmin/amsmedia/downloads/4896_overviewoforielspectrographs.pdf
# In the spectrograph, the frequencies of both beams are dispersed and imaged onto a 2×32 HgCdTe (MCT) array detector (Infrared Associates).
# The pixels are 0.5 mm wide by 1 mm high, and the arrays are separated by a vertical 8 mm
# See Panman, M.R.'s thesis for details, page 18

def processScans(filename, nscans):
    scans = open(filename,'r').read().split('# scan')[1:nscans+1]
    t = np.asarray([float(delay.split()[0]) for delay in scans[0].split('\n')[1:-2]])
    dT = np.dstack([np.vstack([np.array(list(map(float,delay.split()[1:-1:4]))) for delay in scan.split('\n')[1:len(t)+1]]).T for scan in scans])    # Difference transmission
    dT_err = np.dstack([np.vstack([np.array(list(map(float,delay.split()[2::4]))) for delay in scan.split('\n')[1:len(t)+1]]).T for scan in scans]) # Standard deviation
    dA = -np.log10(dT)                   #Difference absorption
    dA_err = abs(dT_err/(dT*np.log(10)))  # Error propagation for log10  
    return t, dA,dA_err





def getPixelArray(grating, centerwave, centerpix, npix):
    reciprocal_dispersion_list = {'150 lines/mm':25.8,'75 lines/mm': 51.7}      # reciprocal dispersion in nm/mm
    reciprocal_dispersion = reciprocal_dispersion_list[grating]
    array = (np.linspace(1,npix,npix)-centerpix)                                    # Center pixel becomes '0' in the 32-element array
    array = 0.5 * array                                                         # Each pixel center is separated by 0.5 mm
    centerwave_nm = 1E7/centerwave
    wavelength_nm = reciprocal_dispersion*array+centerwave_nm
    wavelength_wavenumbers = 1E7/wavelength_nm
    return wavelength_wavenumbers

# This function will load the data file and save it into the 'data' dictionary


def loadData( filename ): 
    data = {}
    lines = open(filename, 'r').read().split('\n')[:-1]
    data['filename'] = ' '.join(lines[2].split()[3:]) # original filename
    data['nscans'] = int(lines[-1].split()[-1]) # n scans completed
    data['npixels'] = int(lines[6].split()[-1]) 
    data['centerwave'] = float(lines[89].split()[-2]) # wavenumbers
    data['grating'] = ' '.join(lines[90].split()[-2:])  # lines/mm
    data['centerpix'] = int(lines[91].split()[-1]) # pixel at center of img
    data['x'] = getPixelArray(data['grating'],data['centerwave'],data['centerpix'], data['npixels']) # the detector array
    data['delays'], data['dA'], data['dA_err'] = processScans(filename, data['nscans'])
    return data



