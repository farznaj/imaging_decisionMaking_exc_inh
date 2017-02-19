"""
crossValidateModel: divides data into training and test datasets. Calls linearSVM.py, which does linear SVM 
using XTrain, and returns percent class loss for XTrain and XTest.

"""

#%%
def crossValidateModel(X, Y, modelFn, **options):
    import numpy as np
    import numpy.random as rng
    from linearSVM import *
    from linearSVR import *
    
    if options.get('kfold'):
        kfold = options.get('kfold');
    else:
        kfold = 10;
        
#    Y = np.squeeze(np.array(Y).astype(int)); # commented so it works for svr too.
        
    if X.shape[0]>len(Y):
        numObservations = len(Y);
        numFeatures = len(X)/numObservations;
        X = np.reshape(np.array(X.astype('float')), (numObservations, numFeatures), order='F');
    
    numObservations, numFeatures = X.shape;
    
    ## %%%%%% shuffle trials to break any dependencies on the sequence of trails 
    shfl = rng.permutation(np.arange(0, numObservations));
    Ys = Y[shfl];
    Xs = X[shfl, :]; 
    
    ## %%%%% divide data to training and testin sets
    YTrain = Ys[np.arange(0, int((kfold-1.)/kfold*numObservations))]; # Take the first 90% of trials as training set
    YTest = Ys[np.arange(int((kfold-1.)/kfold*numObservations), numObservations)]; # Take the last 10% of trials as testing set

    XTrain = Xs[np.arange(0, int((kfold-1.)/kfold*numObservations)), :];
    XTest = Xs[np.arange(int((kfold-1.)/kfold*numObservations), numObservations), :];

    # Fit the classifier
    results = modelFn(XTrain, YTrain, XTest, YTest, **options)
    
    return results;

