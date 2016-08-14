"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2016 Gamaleldin F. Elsayed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gamaleldin F. Elsayed
% 
% logisticRegression.py
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X: features array of size features * observation. This is reshaped to be a 
% matrix obsevations by features
% Y: class labels array of size number o observations
% l: array of size 2 that contains the regularization parameter for l2 at the 
% first entry and l1 at the sencond entry
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#%%
def logisticRegression(X, Y, l, **options):
    #%% imports
    from theano import function
    from theano import shared
    import numpy as np
    from numpy import random as rng
    import matplotlib.pyplot as plt
    from theano import tensor as Tn
    from lasagne.updates import adagrad
    if options.get('plotFigure'):
        plotFigure = options.get('plotFigure');
    else:
        plotFigure = False;
        
    if options.get('verbose'):
        verbose = options.get('verbose');
    else:
        verbose = False;         
    #%% load data
    l = np.array(l).astype(float);
    numObservations = len(Y);
    numFeatures = len(X)/numObservations;
    Y = np.squeeze(np.array(Y).astype('int'));    
    X = np.reshape(np.array(X).astype('float'), (numObservations, numFeatures), order='F');
    # remove any nans    
    mskOutNans = (np.sum(np.isnan(X), axis = 1)+ np.squeeze(np.isnan(Y)))<1 ;
    X = X[mskOutNans, :];
    Y = Y[mskOutNans];    
    numObservations, numFeatures = X.shape;
    Data = (X, Y); # data tuple
    
    #%% declaration of sympolic variables and initializations
    # Declare theano symbolic input/ output variables
    x = Tn.matrix('x'); # sympolic feature variable
    y = Tn.vector('y'); # sympolic label variable
    lambda_l2 = Tn.dscalar('lambda_l2'); # sympolic l2-regularization parameter variable
    lambda_l1 = Tn.dscalar('lambda_l1'); # sympolic l1-regularization parameter variable
    # Declare and initialize theano shared optimization variables
    w = shared(rng.randn(numFeatures), name = 'w'); # initialize the weight vector (w) randomly
    b = shared(np.random.randn(), name = 'b'); # initialize the bias variable (b) randomly
    lps = shared(0.5 * np.random.rand(), name = 'lps'); # initialize the lapse rate variable (lps) randomly between 0 and 1
    
    #%% functions expressions and compilations
    # function sympolic expressions
    XW = Tn.dot(x, w);
    L_Exp = Tn.shape(y); 
    prob1Expression = lps + (1 - 2 * lps)/(1.0 + Tn.exp(- (XW + b))); # expression of logistic function (prob of 1)
    prob0Expression = 1.0-prob1Expression; # expression of logistic function (prob of 0)    
    predictExpression = prob1Expression>0.5;
    logLikelihood_Exp = (y * Tn.log(prob1Expression) + (1.0 - y) * Tn.log(prob0Expression)).sum(); # loglikelihood expression
    costExpression = - logLikelihood_Exp/(L_Exp[0]) + lambda_l2 * (w ** 2).sum() + lambda_l1 * abs(w).sum();   # mean cost across all samples with the regularization
    perClassErExpression = abs(predictExpression - y).sum()/numObservations*100; # percent classification error expression    
    
    # compiling function expressions for speed    
    prob1Fn = function(inputs = [x], outputs = prob1Expression); 
    prob0Fn = function(inputs = [x], outputs = prob0Expression); 
    predictFn = function(inputs = [x], outputs = predictExpression);
    costFn = function(inputs = [x, y, lambda_l2, lambda_l1], outputs = costExpression);
    perClassErFn = function(inputs = [x, y], outputs = perClassErExpression);

    # cost gradient with respect to all parameters
    grad_w, grad_b, grad_lps = Tn.grad(costExpression, [w, b, lps]);

    updates = adagrad([grad_w, grad_b, grad_lps], [w, b, lps], learning_rate=0.1, epsilon=0.001)

    # training function
    trainFn = function(
    inputs = [x, y, lambda_l2, lambda_l1],
    outputs = [costExpression, perClassErExpression, logLikelihood_Exp],
    updates = updates    
    #updates = ((w, w - learnRate*grad_w), (b, b - learnRate*grad_b), (lps, Tn.clip((lps - learnRate*grad_lps) , 0.0, 0.4999999)))
    )

    #%% Training the model
    maxIter = 10000;
    numRepetitions = 3;
    w_0 = [];
    b_0 = [];
    lps_0 = [];
    minCost = np.inf;
    rbest = 0
    for r in range(numRepetitions):
        w_0.append(rng.randn(numFeatures));
        b_0.append(np.random.rand());
        lps_0.append(0.5*np.random.rand());    
        
        w.set_value(w_0[r])
        b.set_value(b_0[r])
        lps.set_value(lps_0[r])        
        b_i = [b.get_value()];
        lps_i = [lps.get_value()];
        cost = [];
        perClassEr = [];
        lklhood_i = [];
        for i in range(int(maxIter)):
            Er1, Er2, lklhood = trainFn(Data[0], Data[1], l[0], l[1]);
            lps.set_value(np.clip(lps.get_value(), 0., 0.5)) # enforce constraint
            lklhood_i.append(lklhood)            
            cost.append(Er1);
            perClassEr.append(Er2);
            b_i.append(b.get_value());
            lps_i.append(lps.get_value());
            if i>500:
                if abs(cost[i-100]-cost[i])<(10.0**-5):  
                    break;
            if verbose:
                print 'iteration %d , objective value %.5f, initial conditions %d out of %d' %(i+1, cost[i],r+1, numRepetitions) 
        if cost[-1]<minCost:
            rbest = r;
            minCost = cost[-1];
            costbest = cost;
            perClassErbest = perClassEr;
            wbest = w.get_value();
            lpsbest = lps.get_value();
            bbest = b.get_value();
            bbest_i = b_i;
            lpsbest_i = lps_i;
            lklhoodBest_i = lklhood_i;          
      
    w_0 = w_0[rbest]
    cost = costbest;
    perClassEr = perClassErbest;
    w.set_value(wbest) ;
    lps.set_value(lpsbest);
    b.set_value(bbest);
    b_i = bbest_i;
    lps_i = lpsbest_i;
    lklhood_i = lklhoodBest_i;          


  
    #%% plot results
    #%%
    msk1 = Data[1]>0
    msk0 = Data[1]<1
    Y1 = Data[1][msk1]
    Y0 = Data[1][msk0]
    Yhat = predictFn(Data[0])
    Yhat1 = Yhat[msk1];
    Yhat0 = Yhat[msk0];
    prob1 = prob1Fn(Data[0])    
    prob1_1 = prob1[msk1]    
    prob1_0 = prob1[msk0]    
    
    if plotFigure:
        
        plt.figure('cost')
        plt.subplot(3,1,1)
        plt.plot(cost)
        plt.xlabel('iteration')
        plt.ylabel('cross-entropy loss')
        plt.subplot(3,1,2)
        plt.plot(perClassEr)
        plt.ylim(0,100)
        plt.xlabel('iteration')
        plt.ylabel('classification error (%)')
        plt.subplot(3,1,3)
        plt.plot(lklhood_i)
        plt.xlabel('iteration')
        plt.ylabel('log likelihood')
    
        plt.figure('prediction')
        plt.plot(np.arange(1, len(Y0)+1), Yhat0, 'ro', label='prediction')
        plt.plot(np.arange(1, len(Y0)+1), prob1_0, 'b.', label='likelihood of success')
        plt.plot(np.arange(len(Y0)+1, len(Yhat)+1), Yhat1, 'ro')
        plt.plot(np.arange(len(Y0)+1, len(Yhat)+1), prob1_1, 'b.')
        plt.plot(np.arange(1, len(Y0)+1), Y0, 'k.', label='data')
        plt.plot(np.arange(len(Y0)+1, len(Yhat)+1), Y1, 'k.')
    
        plt.ylim(-0.1,1.1)
        plt.xticks([0 , 1])
        plt.xlabel('observation')
        plt.ylabel('class label')
        plt.legend()
    
        plt.figure('weights')
        plt.subplot(3, 1, 1)
        plt.plot(w_0, 'g')
        plt.plot(w.get_value(), 'r')
        plt.xlabel('feature number')
        plt.ylabel('feature weight')
        plt.legend(('before optimization', 'after optimization'))
        plt.subplot(3, 1, 2)
        plt.plot(b_i)
        plt.xlabel('iteration')
        plt.ylabel('bias')
        plt.subplot(3, 1, 3)
        plt.plot(lps_i)
        plt.ylim(0., 1.)
        plt.xlabel('iteration')
        plt.ylabel('lapse rate')
        
    #%% save optimization parameters to class
    class optParamsClass:
        cost_per_iter = np.inf;
        perClassEr_per_iter = 100.;
        loglikelihood = -np.inf;
        prediction = [];
        likelihood_trial = [];
        predictFn = [];
        perClassErFn = [];
        probSucessFn = [];
        def description(self):
            return 'object contains optimization parameters'
    
    optParams = optParamsClass();
    optParams.cost_per_iter = np.array(cost);
    optParams.perClassEr_per_iter = np.array(perClassEr);
    optParams.loglikelihood = lklhood_i;
    optParams.prediction = Yhat;
    optParams.likelihood_trial = prob1;
    optParams.predictFn = predictFn;
    optParams.perClassErFn = perClassErFn;
    optParams.probSucessFn = prob1Fn;
    #%% return parameters
    return w.get_value(), b.get_value(), lps.get_value(), perClassEr[-1], cost[-1], optParams



#%% 
def crossValidateLogistic(X, Y, regType, kfold, numSamples):
    import numpy as np
    from logisticRegression import logisticRegression
    import matplotlib.pyplot as plt
    from numpy import random as rng
    from compiler.ast import flatten

    scale = np.sqrt((X**2).mean());
    mskOutNans = (np.sum(np.isnan(X), axis = 1)+ np.squeeze(np.isnan(Y)))<1 ;
    X = X[mskOutNans, :];
    Y = Y[mskOutNans];  
    numObservations, numFeatures = X.shape;
    lvect = flatten([0, list(10**(np.arange(-5, 2,0.5)))]);
    
    l = np.zeros((len(lvect), 2))
    if regType== 'l2':
        l[:, 0] = lvect; # l2-regularization
    elif regType== 'l1':
        l[:, 1] = lvect; # l1-regularization
    else:
        lvect = [0., 0.] # no regularization
        l = [0., 0.]
            
    l = l*scale;
    
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
        for i in range(l.shape[0]):
            w, b, lps, perClassEr, cost, optParams = logisticRegression(np.reshape(XTrain, (np.prod(XTrain.shape)), order = 'F'), YTrain, l[i, :])
            perClassErrorTest[s, i] = optParams.perClassErFn(XTest, YTest);
            perClassErrorTrain[s, i] = optParams.perClassErFn(XTrain, YTrain);
        print 'cross-validating: %.2f %% completed' % ((s+1.)/(numSamples+0.)*100.) 
    
    meanPerClassErrorTrain = np.mean(perClassErrorTrain, axis = 0);
    semPerClassErrorTrain = np.std(perClassErrorTrain, axis = 0)/np.sqrt(numSamples);
    
    meanPerClassErrorTest = np.mean(perClassErrorTest, axis = 0);
    semPerClassErrorTest = np.std(perClassErrorTest, axis = 0)/np.sqrt(numSamples);
    ix = np.argmin(meanPerClassErrorTest);
    l = l[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix]), :];
    lbest = l[-1, :]; # best regularization term based on minError+SE criteria
    ix = np.sum(l==lbest,1)==2
    ##%%%%%% plot coss-validation results
    plt.figure('cross validation')
    
    plt.fill_between(lvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
    plt.fill_between(lvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
    plt.plot(lvect, meanPerClassErrorTrain, 'k', label = 'training')
    plt.plot(lvect, meanPerClassErrorTest, 'r', label = 'validation')
    plt.plot(np.array(lvect)[ix], meanPerClassErrorTest[ix], 'bo')
    plt.xlim([lvect[1], lvect[-1]])
    plt.xscale('log')
    plt.xlabel('regularization parameter')
    plt.ylabel('classification error (%)')
    plt.legend()
    return lbest

            