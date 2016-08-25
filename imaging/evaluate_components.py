# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:40:37 2016

@author: Andrea G with small modifications from farznaj
"""

import scipy  
import numpy
import numpy as np
from scipy.stats import norm    

#%%
def evaluate_components(traces, N=5, robust_std=False):
    """ Define a metric and order components according to the probabilty if some "exceptional events" (like a spike). Suvh probability is defined as the likeihood of observing the actual trace value over N samples given an estimated noise distribution. 
    The function first estimates the noise distribution by considering the dispersion around the mode. This is done only using values lower than the mode. The estimation of the noise std is made robust by using the approximation std=iqr/1.349. 
    Then, the probavility of having N consecutive eventsis estimated. This probability is used to order the components. 

    Parameters
    ----------
    traces: ndarray
        Fluorescence traces 

    N: int
        N number of consecutive events


    Returns
    -------
    idx_components: ndarray
        the components ordered according to the fitness

    fitness: ndarray


    erfc: ndarray
        probability at each time step of observing the N consequtive actual trace values given the distribution of noise

    """
    
   # import pdb
   # pdb.set_trace()
    md = mode_robust(traces, axis=1)
    ff1 = traces - md[:, None]
    # only consider values under the mode to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)
    if robust_std:
        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        Ns = np.round(np.sum(ff1 > 0, 1) * .5)
        iqr_h = np.zeros(traces.shape[0])

        for idx, el in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -Ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        Ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(np.sum(ff1**2, 1) / Ns)
#


    # compute z value
    z = (traces - md[:, None]) / (3 * sd_r[:, None])
    # probability of observing values larger or equal to z given notmal
    # distribution with mean md and std sd_r
    erf = 1 - norm.cdf(z)
    # use logarithm so that multiplication becomes sum
    erf = np.log(erf)
    filt = np.ones(N)
    # moving sum
    erfc = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='full'), axis=1, arr=erf)

    # select the maximum value of such probability for each trace
    fitness = np.min(erfc, 1)

    ordered = np.argsort(fitness)

    idx_components = ordered  # [::-1]# selec only portion of components
    fitness = fitness[idx_components]
    erfc = erfc[idx_components]

    return idx_components, fitness, erfc
#%%


def mode_robust(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """
    if axis is not None:
        fnc = lambda x: mode_robust(x, dtype=dtype)
        dataMode = numpy.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:
                
                #            wMin = data[-1] - data[0]
                wMin = np.inf
                N = data.size / 2 + data.size % 2

                for i in xrange(0, N):
                    w = data[i + N - 1] - data[i]
                    if w < wMin:
                        wMin = w
                        j = i

                return _hsm(data[j:j + N])

        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)

        # The data need to be sorted for this to work
        data = numpy.sort(data)

        # Find the mode
        dataMode = _hsm(data)

    return dataMode
#=======
#               return data[1]
#         else:
##            wMin = data[-1] - data[0]
#            wMin=np.inf
#            N = data.size/2 + data.size%2
#            for i in xrange(0, N):
#               w = data[i+N-1] - data[i]
#               if w < wMin:
#                  wMin = w
#                  j = i
#
#            return _hsm(data[j:j+N])
#
#      data = inputData.ravel()
#      if type(data).__name__ == "MaskedArray":
#         data = data.compressed()
#      if dtype is not None:
#         data = data.astype(dtype)
#
#      # The data need to be sorted for this to work
#      data = numpy.sort(data)
#
#      # Find the mode
#      dataMode = _hsm(data)
#
#   return dataMode
#>>>>>>> 9afa11dd3825fd3ac6298b5bcfb3869a55ce3c68
