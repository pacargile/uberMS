import numpy as np
def ccm_curve(wave,ebv,R_V = 3.1):
    '''
    Calculates the reddening of a flux array with wavelength, wave, in nanometers. 
    Assumes the reddening curve of Cardelli, Clayton, and Mathis (1989 ApJ. 345, 245), 
    and the update for the near-UV given by O'Donnell (1994, ApJ, 422, 158). 
    Adapted from the IDL astronomy library routine, ccm_UNRED. 
     
    Inputs: 
    wave   - wavelength vector in nanometers. 
    ebv    - E(B-V) value, this is but one of many ways to express the reddening. 
    (R_V)  - (optional) specifies the ratio of the total selective extinction: 
     
                   R_V = A_V / E(B-V) 
                    
    Returns: 
    fratio - the ratio of the source flux and the observed flux, such that, to calculate 
             the flux observed after reddening, flux_obs, from the source flux, flux_src, 
             one does the following operation: 
              
                   flux_obs = flux_src / fratio
             
             only valid when flux_obs, flux_src, and fratio are all calculated on the 
             same wavelength grid in the variable, wave. 
              
    '''
    # convert wave in nm to inverse micrometers
    x = 1000./wave
    npts = len(x)
 
    a = np.zeros(npts)
    b = np.zeros(npts)
 
    # Reddening in the IR 
    good = np.where((x > 0.3) & (x < 1.1))[0]
    if (len(good) > 0): 
        a[good] = 0.574*x[good]**(1.61)
        b[good] = -0.527*x[good]**(1.61)
 
    # Reddening in Optical and NIR:
    good = np.where((x >= 1.1) & (x <3.3))[0]
    if (len(good) > 0):
        c1 = np.array([ 1. , 0.104,   -0.609,    0.701,  1.137,-1.718,   -0.827,    1.647, -0.505 ])
        c2 = np.array([ 0.,  1.952,    2.908,   -3.989, -7.985, 11.102,    5.491,  -10.805,  3.347 ])
        y = x[good]-1.82
        a[good] = np.polyval(c1[::-1],y)
        b[good] = np.polyval(c2[::-1],y)
 
    # Reddening in Mid-UV: 
    good = np.where((x>=3.3) & (x<8))[0]
    if (len(good)>0):
        y = x[good]
        F_a = np.zeros(len(good))
        F_b = np.zeros(len(good))
        #   
        good1 = np.where( ( y > 5.9))[0]
        if (len(good1)>0):
            y1 = y[good1]-5.9
            F_a[good1] = -0.04473 * y1**2 - 0.009779 * y1**3
            F_b[good1] =   0.2130 * y1**2 +   0.1207 * y1**3
         
        a[good] =  1.752 - 0.316 * y - ( 0.104 / ( (y - 4.67)**2 + 0.341 ) ) + F_a
        b[good] = -3.090 + 1.825 * y + ( 1.206 / ( (y - 4.62)**2 + 0.263 ) ) + F_b
     
    # Reddening in Far-UV:
    good = np.where((x >= 8.) & (x<=11.))[0]
    if (len(good)>0): 
        y = x[good] - 8.
        c1 = np.array([ -1.073, -0.628,  0.137, -0.070 ])
        c2 =  np.array([ 13.670,  4.257, -0.420,  0.374 ])
        a[good] = np.polyval(c1[::-1],y)
        b[good] = np.polyval(c2[::-1],y)
 
    A_V = R_V * ebv 
    A_lambda = A_V * (a + b / R_V) 
     
    return 10.**(0.4*A_lambda)