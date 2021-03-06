# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:00:25 2016

@author: gamalamin
"""

#%% main function
import numpy as np
import scipy
import matplotlib.pyplot as plt
from lassoLinearSVM import *
#%% load data
#dirName = '//sonas-hs.cshl.edu/churchland/data/fni17/imaging/151029/151029_003_ch2-PnevPanResults-160426-191859.mat';
dirName = '/Users/gamalamin/git_local_repository/Farzaneh/XY.mat';
#dirName = '/home/farznaj/Shares/Churchland/data/fni17/imaging/151022/XY_fni17_151022 XY_lassoSVM.mat';
XY = scipy.io.loadmat(dirName, variable_names=['X', 'Y']);
X = XY.pop('X')
Y = XY.pop('Y')
#%% feature normalization and scale
meanX = np.mean(X, axis = 0);
stdX = np.std(X, axis = 0);
X = (X-meanX)/stdX;

X = np.reshape(X, (X.shape[0]*X.shape[1], 1), order='F');#%% do cross validation and pick the regularization parameter
#%%
kfold = 10
numSamples = 2
lbest = crossValidateSVM(X, Y, 'l2', kfold, numSamples)

#%% run lasso SVM
wl1_shuff, b_shuff, wl1, b, bestCVerror, bestc, linear_svm, meanCVerror, semCVerror, c, meanX, stdX = lassoLinearSVM(X, Y, 10)

#%% compare to l2
from sklearn import svm
linear_svm = svm.LinearSVC(loss='squared_hinge', penalty='l2', dual=False)
linear_svm.fit(np.reshape(X, (len(Y), len(X)/len(Y)), order='F'), Y)
wl2 =  np.squeeze(linear_svm.coef_);

#%%
plt.figure('Weights')
plt.plot(np.sort(np.abs(wl1))[::-1], 'r')
plt.plot(np.sort(np.abs(wl2))[::-1], 'g')
plt.ylabel('coefficient')
plt.xlabel('neuron')
plt.legend(('lasso', 'regular'))
