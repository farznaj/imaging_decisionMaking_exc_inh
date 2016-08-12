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
from numpy import random as rng
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
#%% do cross validation and pick the regularization
scale = np.sqrt((X**2).mean());
mskOutNans = (np.sum(np.isnan(X), axis = 1)+ np.squeeze(np.isnan(Y)))<1 ;
X = X[mskOutNans, :];
Y = Y[mskOutNans];  
numObservations, numFeatures = X.shape;
kfold = 10
lvect = 10**np.array(np.arange(-5, 1,0.5))
l = np.zeros((len(lvect)+1, 2))
l[1:, 0] = lvect; # l2-regularization
l[1:, 1] = lvect; # l1-regularization
l = l*scale;
numSamples = 1;  # number of samples for each regularization value (the more the better the estimate of the cross-validation error yet the slower the algorithm)

perClassErrorTest = np.nan+np.ones((numSamples, l.shape[0]));
perClassErrorTrain = np.nan+np.ones((numSamples, l.shape[0]));

for s in range(numSamples):
    ## %%%%%% shuffle trials to break any dependencies on the sequence of trails 
    shfl = rng.permutation(np.arange(0, numObservations));
    Ys = Y[shfl];
    Xs = X[shfl, :]; 
        
    ## %%%%% divide data to training and testin sets
    YTrain = Ys[range(int((kfold-1.)/kfold*numObservations))];
    YTest = Ys[np.arange(int((kfold-1.)/kfold*numObservations), numObservations)];
        
    XTrain = Xs[range(int((kfold-1.)/kfold*numObservations)), :];
    XTest = Xs[np.arange(int((kfold-1.)/kfold*numObservations), numObservations), :];
    ## %%%%% loop over the possible regularization values
    for i in range(len(lvect)):
        w, b, lps, perClassEr, cost, optParams = logisticRegression(np.reshape(XTrain, (np.prod(XTrain.shape)), order = 'F'), YTrain, l[i, :])
        perClassErrorTest[s, i] = optParams.perClassErFn(XTest, YTest);
        perClassErrorTest[s, i] = optParams.perClassErFn(XTrain, YTrain);



#%%
scale = np.sqrt((X**2).mean()).astype('float');
l = scale * np.array([0. , 0.]);
w, b, lps, perClassEr, cost, optParams = logisticRegression(np.reshape(X, (np.prod(X.shape)), order = 'F'), Y, l)

scio.savemat('logisticResults.mat', {'w':w, 'b': b, 'lps':lps, 'perClassEr': perClassEr, 'cost':cost, 'optParams':optParams})