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

def logisticRegression(X, Y, l):
    #%% imports
    from theano import function
    from theano import shared
    import numpy as np
    from numpy import random as rng
    import matplotlib.pyplot as plt
    from theano import tensor as Tn
    
    #%% load data
    l = np.array(l).astype(float);
    numObservations = len(Y);
    numFeatures = len(X)/numObservations;
    Y = np.squeeze(np.array(Y).astype('int'));    
    X = np.reshape(np.array(X).astype('float'), (numObservations, numFeatures), order='F');
    scale = np.sqrt((np.reshape(X, (numObservations*numFeatures), order = 'F')**2).mean());
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
    learnRate = shared(scale, name = 'learnRate')
    
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
    
    # training function
    trainFn = function(
    inputs = [x, y, lambda_l2, lambda_l1],
    outputs = [costExpression, perClassErExpression, logLikelihood_Exp],
    updates = ((w, w - learnRate*grad_w), (b, b - learnRate*grad_b), (lps, Tn.clip((lps - learnRate*grad_lps) , 0.0, 0.4999999)))
    )

    #%% Training the model
    maxIter = 50000;
    numRepetitions = 4;
    w_0 = [];
    b_0 = [];
    lps_0 = [];
    minCost = np.inf;
    for r in range(numRepetitions):
        w_0.append(rng.randn(numFeatures));
        b_0.append(np.random.rand());
        lps_0.append(0.5*np.random.rand());    
        
        w.set_value(w_0[r])
        b.set_value(b_0[r])
        lps.set_value(lps_0[r])        
        learnRate.set_value(1.0*scale)        
        b_i = [b.get_value()];
        lps_i = [lps.get_value()];
        learnRate_i = [learnRate.get_value()];
        cost = [];
        perClassEr = [];
        lklhood_i = [];
        for i in range(int(maxIter/100.)):
            Er1, Er2, lklhood = trainFn(Data[0], Data[1], l[0], l[1]);
            lklhood_i.append(lklhood)            
            cost.append(Er1);
            perClassEr.append(Er2);
            b_i.append(b.get_value());
            lps_i.append(lps.get_value());
            learnRate_i.append(learnRate.get_value());
            if  np.isnan(w.get_value()).sum()>0 or np.isnan(lps.get_value()).sum()>0 or np.isnan(b.get_value()).sum()>0:   
                learnRate.set_value(learnRate.get_value()/2.0) 
                w.set_value(w_0[r]);
                b.set_value(b_0[r]);
                lps.set_value(lps_0[r]);
            if i>500:
                if abs(cost[i-100]-cost[i])<(10.0**-3):  
                    break;
                    
        if cost[-1]<minCost:
            rbest = r;
            minCost = cost[-1];
            costbest = cost;
            perClassErbest = perClassEr;
            wbest = w.get_value();
            lpsbest = lps.get_value();
            bbest = b.get_value();
            learnRatebest_i = learnRate_i;
            bbest_i = b_i;
            lpsbest_i = lps_i;
            lklhoodBest_i = lklhood_i;          
      



    

       
    
    w_0 = w_0[rbest]
    cost = costbest;
    perClassEr = perClassErbest;
    w.set_value(wbest) ;
    lps.set_value(lpsbest);
    b.set_value(bbest);
    learnRate_i = learnRatebest_i;
    b_i = bbest_i;
    lps_i = lpsbest_i;
    lklhood_i = lklhoodBest_i;          

    for i in range(int(95.*maxIter/100.)):
        Er1, Er2, lklhood = trainFn(Data[0], Data[1], l[0], l[1]);
        lklhood_i.append(lklhood)            
        cost.append(Er1);
        perClassEr.append(Er2);
        b_i.append(b.get_value());
        lps_i.append(lps.get_value());
        learnRate_i.append(learnRate.get_value());
        if  np.isnan(w.get_value()).sum()>0 or np.isnan(lps.get_value()).sum()>0 or np.isnan(b.get_value()).sum()>0:   
            learnRate.set_value(learnRate.get_value()/2.0) 
            w.set_value(w_0);
            b.set_value(b_0);
            lps.set_value(lps_0);
        if i>500:
            if abs(cost[i-100]-cost[i])<(10.0**-6):  
                break;
                
    ## pruning the values with smaller learning rate
    learnRate.set_value(learnRate.get_value()/10.0) 
    for i in range(int(4.*maxIter/100.)):
        Er1, Er2 ,lklhood = trainFn(Data[0], Data[1], l[0], l[1]);
        lklhood_i.append(lklhood)            
        cost.append(Er1);
        perClassEr.append(Er2);
        b_i.append(b.get_value());
        lps_i.append(lps.get_value());
        learnRate_i.append(learnRate.get_value());
        if  np.isnan(w.get_value()).sum()>0 or np.isnan(lps.get_value()).sum()>0 or np.isnan(b.get_value()).sum()>0:   
            learnRate.set_value(learnRate.get_value()/2.0) 
            w.set_value(w_0);
            b.set_value(b_0);
            lps.set_value(lps_0);
        if i>500:
            if abs(cost[i-100]-cost[i])<(10.0**-6):  
                break;

       
       
  
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
        def description(self):
            return 'object contains optimization parameters'
    
    optParams = optParamsClass();
    optParams.cost_per_iter = np.array(cost);
    optParams.perClassEr_per_iter = np.array(perClassEr);
    optParams.loglikelihood = lklhood_i;
    optParams.prediction = Yhat;
    optParams.likelihood_trial = prob1;
    #%% return parameters
    return w.get_value(), b.get_value(), lps.get_value(), perClassEr[-1], cost[-1], optParams

                   