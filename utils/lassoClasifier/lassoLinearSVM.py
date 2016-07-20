# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:41:20 2016

@author: gamalamin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:54:46 2016

@author: gamalamin
"""
# X and Y are two vectors
def lassoLinearSVM(X, Y, kfold):
    #%% initialization ans imports
    import numpy as np
    import numpy.random as rng
    from sklearn import svm
    import matplotlib.pyplot as plt
    from scipy import stats
    
    
    Y = np.squeeze(np.array(Y).astype(int));
    L = len(Y);
    N = len(X)/L;
    kfold = int(kfold);    
    
    X = np.reshape(np.array(X.astype('float')), (L, N), order='F');

    #%% feature normalization and scale
    meanX = np.mean(X, axis = 0);
    stdX = np.std(X, axis = 0);
    X = (X-meanX)/stdX;
    
    #%% define functions
    def optimizeSVMReg(X, Y, kfold):
        numSamples = 100;
        c = 10**np.array(np.arange(-5, 3, 0.1))
        CVerror = np.ones((len(c), numSamples))
        for i in range(len(c)):
            for j in range(numSamples):
                CVerror[i, j] = CVSVM(X, Y, c[i], kfold);
        meanCVerror = np.mean(CVerror, axis = 1);
        semCVerror = stats.sem(CVerror, axis = 1);    
        ix = np.argmin(meanCVerror)
       # best error using standard error criteria
        msk = meanCVerror<=meanCVerror[ix]+semCVerror[ix];    
        bestc = c[msk][0]
        bestCVerror = meanCVerror[msk][0]
        plt.figure()
        plt.errorbar(c, meanCVerror,  yerr=semCVerror, color = 'k')
        plt.plot(bestc, bestCVerror, 'ro')
        plt.xscale('log')
        plt.xlabel('inverse lasso regularization parameter (c)')    
        plt.ylabel('cross-validation error')    
        return bestc, bestCVerror, meanCVerror, semCVerror, c
       
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
        
    #%% identify the best reg parameter using 1SE criteria
    bestc, bestCVerror, meanCVerror, semCVerror, c = optimizeSVMReg(X, Y, kfold)
    
    #%% run the linear SVM classifer with best regularization parameter
    # Create a classifier: a support vector classifier
    linear_svm = svm.LinearSVC(C = bestc, loss='squared_hinge', penalty='l1', dual=False)    
    linear_svm.fit(X, Y)
    wl1 =  np.squeeze(linear_svm.coef_);
    b = linear_svm.intercept_;
    
    return wl1, b, bestCVerror, bestc, linear_svm, meanCVerror, semCVerror, c, meanX, stdX
