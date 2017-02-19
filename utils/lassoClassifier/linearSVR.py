"""
Fits SVR using XTrain, and returns percent class loss for XTrain and XTest
"""
#%%
def linearSVR(XTrain, YTrain, XTest, YTest, **options):
    
    import numpy as np
    from sklearn.svm import LinearSVR
#    from sklearn import svm
#    linear_svr = [];
    # Create a classifier: a support vector classifier
    '''
    if options.get('l1'):
        l1 = options.get('l1');
        #print 'running l1 svm classification\r' 
        linear_svr = svm.LinearSVR(C = l1, loss='squared_hinge', penalty='l1', dual=False)
    elif options.get('l2'):
        l2 = options.get('l2');        
        #print 'running l2 svm classification\r' 
        linear_svr = svm.LinearSVR(C = l2, loss='squared_hinge', penalty='l2', dual=True)
        '''        
    l = options.get('l');   
    linear_svr = LinearSVR(C=l, epsilon=0.0, dual = True, tol = 1e-9, fit_intercept = True)        
    linear_svr.fit(XTrain, np.squeeze(YTrain))    


    #%%
#    def perClassError_sr(Y,Yhat,eps0=10**-5):    
#        ce = np.mean(np.logical_and(abs(Y-Yhat) > eps0 , ~np.isnan(Yhat - Y)))*100
#        return ce
    def perClassError_sr(y,yhat):
        err = np.linalg.norm(yhat - y)**2
        maxerr = np.linalg.norm(y+1e-10)**2
#        err = (np.linalg.norm(yhat - y)**2)/len(y)
#        maxerr = np.linalg.norm(y)**2
#        ce = err
        ce = err/ maxerr    
    #    ce = np.linalg.norm(yhat - y)**2 / len(y)
        return ce
    
    perClassErrorTest = perClassError_sr(YTest, linear_svr.predict(XTest));
    perClassErrorTrain = perClassError_sr(YTrain, linear_svr.predict(XTrain));
    
    
    #%%
    class summaryClass:
        perClassErrorTrain = [];
        perClassErrorTest = [];
        model = [];
        XTest = []
        XTrain = []
        YTest = []
        YTrain = []
        
    summary = summaryClass();
    summary.perClassErrorTrain = perClassErrorTrain;
    summary.perClassErrorTest = perClassErrorTest;
    summary.model = linear_svr;
    summary.XTest = XTest
    summary.XTrain = XTrain
    summary.YTest = YTest
    summary.YTrain = YTrain
    
    return summary

#    np.mean()
    
    