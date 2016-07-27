# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:18:32 2016

@author: gamalamin
"""
#%%
import scipy
import numpy as np
from logisticRegression import *
import matplotlib.pyplot as plt
plt.close('all')
#%% data

#dirname = '/Users/gamalamin/git_local_repository/Farzaneh/utils/logisticRegression/XYbehaviour.mat';
dirname = 'C:/Users/fnajafi/Documents/trial_history/utils/logisticRegression/XYbehaviour.mat';
XY = scipy.io.loadmat(dirname, variable_names=['X', 'Y']);

X = XY.pop('X')[:, 1:]
Y = XY.pop('Y')
X = np.reshape(X, (X.shape[0]*X.shape[1]), order = 'F');

#%% random data
"""
numObservations = 200;
numFeatures = 300; # number of features
X = np.random.randn((numObservations*numFeatures))
Y = np.random.randint(0, high=2, size = numObservations)
"""
#%% 
l = [0., 0.]
wVect = []
for i in range(1):
    w, b, lps, perClassEr, cost, optParams = logisticRegression(X, Y, l)
    wVect.append(w)
    