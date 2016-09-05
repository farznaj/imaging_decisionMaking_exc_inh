"""
Fits SVM using XTrain, and returns percnet class loss for XTrain and XTest
"""
#%%
def linearSVM(XTrain, YTrain, XTest, YTest, options):
    import numpy as np
    from sklearn import svm
    linear_svm = [];
    # Create a classifier: a support vector classifier
    if options.get('l1'):
        l1 = options.get('l1');
        #print 'running l1 svm classification\r' 
        linear_svm = svm.LinearSVC(C = l1, loss='squared_hinge', penalty='l1', dual=False)
    elif options.get('l2'):
        l2 = options.get('l2');        
        #print 'running l2 svm classification\r' 
        linear_svm = svm.LinearSVC(C = l2, loss='squared_hinge', penalty='l2', dual=True)
        
    linear_svm.fit(XTrain, np.squeeze(YTrain))    

    def perClassError(Y, Yhat):
        import numpy as np
        perClassEr = sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
        return perClassEr
    
    perClassErrorTest = perClassError(YTest, linear_svm.predict(XTest));
    perClassErrorTrain = perClassError(YTrain, linear_svm.predict(XTrain));
    
    class summaryClass:
        perClassErrorTrain = [];
        perClassErrorTest = [];
        model = [];
    summary = summaryClass();
    summary.perClassErrorTrain = perClassErrorTrain;
    summary.perClassErrorTest = perClassErrorTest;
    summary.model = linear_svm;
    return summary

    np.mean()