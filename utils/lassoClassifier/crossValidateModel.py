"""
crossValidateModel: divides data into training and test datasets. Calls linearSVM.py, which does linear SVM 
using XTrain, and returns percent class loss for XTrain and XTest.

"""
# summary,_ =  crossValidateModel(X[ifr,:,:].transpose(), Y, linearSVM, kfold = kfold, l1 = cvect[i], shflTrs = shuffleTrs)
#%%
def crossValidateModel(X, Y, modelFn, **options):
    import numpy as np
    import numpy.random as rng
    from linearSVM import linearSVM
    from linearSVR import linearSVR
    
    if options.get('kfold'):
        kfold = options.get('kfold')
    else:
        kfold = 10;

    if np.logical_or(options.get('shflTrs'), options.get('shflTrs')==0):
        shflTrs = options.get('shflTrs')
    else:
        shflTrs = True
#    print shflTrs
        
#    Y = np.squeeze(np.array(Y).astype(int)); # commented so it works for svr too.
        
    if X.shape[0]>len(Y):
        numObservations = len(Y);
        numFeatures = len(X)/numObservations;
        X = np.reshape(np.array(X.astype('float')), (numObservations, numFeatures), order='F');
    
    numObservations, numFeatures = X.shape # trials x neurons
    
    
    ## %%%%%
    cls = [0]
    while len(cls)<2: # make sure both classes exist in YTrain
    
        if shflTrs==1: # shuffle trials to break any dependencies on the sequence of trails; Also since we take the first 90% of trials as training and the last 10% as testing, for each run of this code we want to make sure we use different sets of trials as testing and training.
            print 'shuffling trials in crossValidateModel'
            shfl = rng.permutation(np.arange(0, numObservations))
            Ys = Y[shfl]
            Xs = X[shfl, :]
            testTrInds = shfl[np.arange(int((kfold-1.)/kfold*numObservations), numObservations)] # index of testing trials (that will be used in svm below)
        else:
            shfl = np.arange(0, numObservations)
            Ys = Y
            Xs = X
            
        
        ## %%%%% divide data to training and testing sets
        YTrain = Ys[np.arange(0, int((kfold-1.)/kfold*numObservations))] # Take the first 90% of trials as training set       
        
        cls = np.unique(YTrain)        
#        print cls
    
    
    if len(cls)==2:        
        YTest = Ys[np.arange(int((kfold-1.)/kfold*numObservations), numObservations)] # Take the last 10% of trials as testing set
    
        XTrain = Xs[np.arange(0, int((kfold-1.)/kfold*numObservations)), :]
        XTest = Xs[np.arange(int((kfold-1.)/kfold*numObservations), numObservations), :]
    
    
        # Fit the classifier
        results = modelFn(XTrain, YTrain, XTest, YTest, **options)
        
        return results, shfl # shfl includes the index of trials in X whose first 90% are used for training and the last 10% are used for testing.

