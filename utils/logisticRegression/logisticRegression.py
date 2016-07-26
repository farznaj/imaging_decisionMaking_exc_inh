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
    lps = shared(np.random.rand(), name = 'lps'); # initialize the lapse rate variable (lps) randomly between 0 and 1
    learnRate = shared(0.1*scale, name = 'learnRate')
    #%% functions expressions and compilations
    # function sympolic expressions
    prob1Expression = lps/(1.0 + Tn.exp(- Tn.dot(x, w) + b)); # expression of logistic function (prob of 1)
    prob0Expression = 1.0-prob1Expression; # expression of logistic function (prob of 0)    
    predictExpression = prob1Expression>0.5;
    costExpression1 = - y * Tn.log(prob1Expression) - (1.0 - y) * Tn.log(prob0Expression); # cost function of one sample
    costExpression = costExpression1.mean() + scale * lambda_l2 * (w ** 2).sum() + scale * lambda_l1 * abs(w).sum();   # mean cost across all samples with the regularization
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
    outputs = [costExpression, perClassErExpression],
    updates = ((w, w - learnRate*grad_w), (b, b - learnRate*grad_b), (lps, Tn.clip((lps - learnRate*grad_lps) , 0.0, 0.999999)))
    )

    #%% Training the model
    maxIter = 50000;
    w0 = w.get_value();
    b0 = b.get_value();
    lps0 = lps.get_value();    
    cost = [];
    perClassEr = [];
    b_i = [b.get_value()];
    lps_i = [lps.get_value()];
    learnRate_i = [learnRate.get_value()];
    for i in range(maxIter):
        Er1, Er2 = trainFn(Data[0], Data[1], l[0], l[1]);
        cost.append(Er1);
        perClassEr.append(Er2);
        b_i.append(b.get_value());
        lps_i.append(lps.get_value());
        learnRate_i.append(learnRate.get_value());

        if  np.isnan(w.get_value()).sum()>0:   
            learnRate.set_value(learnRate.get_value()/2.0) 
            w.set_value(w0);
            b.set_value(b0);
            lps.set_value(lps0);
        if i>500:
            if abs(cost[i-100]-cost[i])<(10.0**-10):
                break;
    #%% plot results
    #%%
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
    plt.plot(learnRate_i)
    plt.xlabel('iteration')
    plt.ylabel('learning rate')

    plt.figure('prediction')
    plt.plot(Data[1], 'go')
    plt.plot(predictFn(Data[0]), 'rx')
    plt.ylim(-0.1,1.1)
    plt.xticks([0 , 1])
    plt.xlabel('observation')
    plt.ylabel('class label')
    plt.legend(('real observations', 'predicted observations'))

    plt.figure('weights')
    plt.subplot(3, 1, 1)
    plt.plot(w0, 'g')
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
    
        def description(self):
            return 'object contains optimization parameters'
    
    optParams = optParamsClass();
    optParams.cost_per_iter = np.array(cost);
    optParams.perClassEr_per_iter = np.array(perClassEr);
    #%% return parameters
    return w.get_value(), b.get_value(), lps.get_value(), perClassEr[-1], cost[-1], optParams

                   