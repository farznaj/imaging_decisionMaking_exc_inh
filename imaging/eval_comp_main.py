# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:52:25 2016

@author: farznaj
"""

#%% Specify file you wish to analyze
mousename = 'fni17'
imagingFolder = '151101'
mdfFileNumber = [1] 

#%% Set pnevFileName 
pnev2load = []; #[3] # which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
signalCh = [2] # since gcamp is channel 2, should be always 2.

from setImagingAnalysisNamesP import *

imfilename, pnevFileName = setImagingAnalysisNamesP(mousename, imagingFolder, mdfFileNumber, signalCh=signalCh, pnev2load=pnev2load)
print pnevFileName

#%% Do imports
import scipy     
from scipy import io
import numpy as np
import os
import h5py
#import numpy.matlib
from evaluate_components import *

#%% Load C and YrA to set traces
f = h5py.File(pnevFileName)
C = np.array(np.transpose(f.get('C')))
YrA = np.array(np.transpose(f.get('YrA')))
'''
a = scipy.io.loadmat(pnevFileName, variable_names=['C', 'YrA']) #.pop('C').astype('float');  
C = a.pop('C')
YrA = a.pop('YrA')
'''
#a = scipy.io.loadmat(dirname, variable_names=['traces'])
#a = traces.pop('traces').astype('float')
traces = C + YrA;
np.shape(traces)

#%% Run the function evaluate_components
idx_components, fitness, erfc = evaluate_components(traces, 5, 0)
'''
traces = np.diff(traces, axis=1)
np.shape(traces)
idx_components, fitness, erfc = evaluate_components(traces, 5, 0)
'''

# Plot
from matplotlib import pyplot as plt
plt.figure
plt.subplot(2,1,1)
plt.plot(fitness)
plt.ylabel('fitness')
plt.subplot(2,1,2)
plt.plot(idx_components)
plt.xlabel('new index')
plt.ylabel('old index')
print np.shape(erfc)


#%% Save results to mat file named "more_pnevFileName"
fname = os.path.join(os.path.dirname(pnevFileName), 'more_'+os.path.basename(pnevFileName))

io.savemat(fname,{'idx_components':idx_components, 'fitness':fitness, 'erfc':erfc})
#with open(fname,'ab') as f:
#    io.savemat(fname,{'idx_componentsD':idx_components, 'fitnessD':fitness, 'erfcD':erfc})
  
#%% Append (didn't work!)
'''
if os.path.isfile(fname): # append doesn't work! it only works if the mat file is created here, but not in matlab.
    with open(fname,'ab') as f:
        io.savemat(f, {'idx_components':idx_components, 'fitness':fitness, 'erfc':erfc}) # append
else:
    io.savemat(fname,{'idx_components':idx_components, 'fitness':fitness, 'erfc':erfc})
'''
'''
# load more_pnev and add it to dictionary for saving
a = io.loadmat(fname, variable_names=['mask', 'CC'])
CC = a.pop('CC')
mask = a.pop('mask')
io.savemat(fname, {'idx_components':idx_components, 'fitness':fitness, 'erfc':erfc, 'CC':CC, 'mask':mask})
'''
