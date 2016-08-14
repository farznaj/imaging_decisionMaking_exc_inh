# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:18:32 2016

@author: gamalamin
"""
#%%
import scipy.io as scio
import numpy as np
from logisticRegression import *
import matplotlib.pyplot as plt

plt.close('all')
#%% load data

dirname = '/Users/gamalamin/git_local_repository/Farzaneh/utils/logisticRegression/XYbehaviour.mat';
#dirname = 'C:/Users/fnajafi/Documents/trial_history/utils/logisticRegression/XYbehaviour.mat';
#dirname = '/media/farznaj/OS/Users/fnajafi/Documents/trial_history/utils/logisticRegression/XYbehaviour.mat';
XY = scio.loadmat(dirname, variable_names=['X', 'Y']);
X = XY.pop('X')[:, 1:]
Y = np.squeeze(XY.pop('Y'));

#%% random data
"""
numObservations = 200;
numFeatures = 300; # number of features
X = np.random.randn((numObservations*numFeatures))
Y = np.random.randint(0, high=2, size = numObservations)
"""
#%% do cross validation and pick the regularization parameter

kfold = 10
numSamples = 100
lbest = crossValidateLogistic(X, Y, 'l2', kfold, numSamples)

#%%
#w, b, lps, perClassEr, cost, optParams = logisticRegression(np.reshape(X, (np.prod(X.shape)), order = 'F'), Y, [0.,0.], plotFigure = True, verbose = True)
w, b, lps, perClassEr, cost, optParams = logisticRegression(np.reshape(X, (np.prod(X.shape)), order = 'F'), Y, lbest, plotFigure = True, verbose = True)
scio.savemat('logisticResults.mat', {'w':w, 'b': b, 'lps':lps, 'perClassEr': perClassEr, 'cost':cost})