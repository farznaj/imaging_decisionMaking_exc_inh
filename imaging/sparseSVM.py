# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:54:46 2016

@author: gamalamin
"""
import numpy as np
import numpy.random as rng
import sklearn as sk
import scipy.linalg as la
import scipy
from sklearn import svm
import matplotlib.pyplot as plt
from time import time

#%% load data
XY = scipy.io.loadmat('/Users/gamalamin/git_local_repository/Farzaneh/XY.mat', variable_names=['X', 'Y']);
X = XY.pop('X')
Y = XY.pop('Y')
#%% feature normalization and scale
meanX = np.mean(X, axis = 0);
stdX = np.std(X, axis = 0);
X = (X-meanX)/stdX
#%%
data_train = X
targets_train = np.squeeze(Y)
data_test = X
targets_test = np.squeeze(Y)
#%% create svm
# Create a classifier: a support vector classifier
linear_svm = svm.LinearSVC(loss='squared_hinge', penalty='l2', dual=False)

linear_svm_time = time()
linear_svm.fit(data_train, targets_train)
linear_svm_scorel2 = linear_svm.score(data_test, targets_test)
linear_svm_time = time() - linear_svm_time
wl2 =  np.squeeze(linear_svm.coef_);

# Create a classifier: a support vector classifier
linear_svm = svm.LinearSVC(C = 1000, loss='squared_hinge', penalty='l1', dual=False)

linear_svm_time = time()
linear_svm.fit(data_train, targets_train)
linear_svm_scorel1 = linear_svm.score(data_test, targets_test)
linear_svm_time = time() - linear_svm_time
wl1 =  np.squeeze(linear_svm.coef_);

#%%
plt.figure('Weights')
plt.plot(np.sort(np.abs(wl1))[::-1], 'r')
plt.plot(np.sort(np.abs(wl2))[::-1], 'g')
#%%
plt.figure()
plt.plot(wl1, 'r')
plt.plot(wl2, 'g')

#%%
def CVSVM(X, Y, c, kfold):
    L = len(np.squeeze(Y));
    ix = rng.permutation(np.arange(L))
    X = X[ix, :];
    Y = Y[ix];
    XCV = X[range(0, L/kfold), :]
    YCV = Y[range(0, L/kfold)]
    
    XTrain = X[range(L/kfold, L), :]
    YTrain = Y[range(L/kfold, L)]
    
    # Create a classifier: a support vector classifier
    linear_svm = svm.LinearSVC(C = c, loss='squared_hinge', penalty='l1', dual=False)
    linear_svm.fit(XTrain, np.squeeze(YTrain))
    CVerror = predictError(YCV, linear_svm.predict(XCV));
    return CVerror
    
def predictError(Y, Yhat):
    error = sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)
    return error
    
#%%

def optimizeSVMReg(X, Y, kfold):
    numSamples = 1000;
    c = 10**np.array(np.arange(-5, 3, 0.5))
    CVerror = np.ones((len(c), numSamples))
    for i in range(len(c)):
        for j in range(numSamples):
            CVerror[i, j] = CVSVM(X, Y, c[i], kfold);
    CVerror = np.mean(CVerror, axis = 1)
    ix = np.argmin(CVerror)
    bestc = c[ix]
    plt.figure()
    plt.semilogx(c, CVerror, 'k')
    plt.semilogx(bestc, CVerror[ix], 'ro')
    return bestc, CVerror
    
    
#%%
bestc, CVerror = optimizeSVMReg(X, Y, 10)
            
#%%
#%% create svm
# Create a classifier: a support vector classifier
linear_svm = svm.LinearSVC(loss='squared_hinge', penalty='l2', dual=False)

linear_svm_time = time()
linear_svm.fit(data_train, targets_train)
linear_svm_scorel2 = linear_svm.score(data_test, targets_test)
linear_svm_time = time() - linear_svm_time
wl2 =  np.squeeze(linear_svm.coef_);

# Create a classifier: a support vector classifier
linear_svm = svm.LinearSVC(C = bestc, loss='squared_hinge', penalty='l1', dual=False)

linear_svm_time = time()
linear_svm.fit(data_train, targets_train)
linear_svm_scorel1 = linear_svm.score(data_test, targets_test)
linear_svm_time = time() - linear_svm_time
wl1 =  np.squeeze(linear_svm.coef_);

#%%
plt.figure('Weights')
plt.plot(np.sort(np.abs(wl1))[::-1], 'r')
plt.plot(np.sort(np.abs(wl2))[::-1], 'g')