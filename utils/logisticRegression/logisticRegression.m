%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2016 Gamaleldin F. Elsayed and Farzaneh Najafi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gamaleldin F. Elsayed
% 
% lassoLinearSVM.m
%
% This function performs linear SVM classification with l1 penalty. The 
% l1 regularization parameter is obtained by kfold cross validation with 1SE
% criteria. This function interfaces and calls a python function 
% (lassoLinearSVM.py), which does all the heavy-duty work.
%
% Inputs:
%   X: features matrix of size (trials x parameters).
%   Y: labels vector of size (trials x 1).
%   kfold: first element; specifies the number of folds for cross
%   validation (if 0 no regularization is performed)
%   regType: if 'l2' l2-regularization is performed, if 'l1' then
%   l1-regularization is performed.
% Outputs:
%   Summary: structure with the following fields
%   	.Model: the scalar slope of the regression model
%           .Beta: coefficients of size (neuorns x 1).
%           .Bias: bias component.
%           .lps: lapse rate component.
%           .l2_regularization: l2-regularization parameter used.
%           .l1_regularization: l1-regularization parameter used.
%           .perClassEr: percent classification error
%           .cost: optimization cost-function value.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Summary] = logisticRegression(X, Y, kfold, regType)
% add path to python. You need to be in the directory of lassoLinearSVM
% functions.
if count(py.sys.path,'') == 0
insert(py.sys.path,int32(0),'');
end
%% perform cross-validation to choose the best regularization parameter
scale = sqrt(mean(X(:).^2));
numTrials = length(Y);  

if kfold>0
    lvect = [0; 10.^(-5:0.5:1)']; % regularization values to choose from, with no regularization option included in the first element
    if strcmpi(regType, 'l2')
        l = [lvect zeros(length(lvect), 1)]; % l2-regularization
    elseif strcmpi(regType, 'l1')
        l = [zeros(length(lvect), 1) lvect]; % l1-regularization
    end
    numSamples = 100;  % number of samples for each regularization value (the more the better the estimate of the cross-validation error yet the slower the algorithm)

    perClassErrorTest = nan(numSamples, length(lvect));
    perClassErrorTrain = nan(numSamples, length(lvect));
    for s = 1:numSamples
        %%%%%% shuffle trials to break any dependencies on the sequence of trails 
        shfl = randperm(numTrials);
        Ys = Y(shfl);
        Xs = X(shfl, :); 
        
        %%%%% divide data to training and testin sets
        YTrain = Ys(1:floor((kfold-1)/kfold*numTrials));
        YTest = Ys(floor((kfold-1)/kfold*numTrials)+1:numTrials);
        
        XTrain = Xs(1:floor((kfold-1)/kfold*numTrials), :);
        XTest = Xs(floor((kfold-1)/kfold*numTrials)+1:numTrials, :);
        
        %%%%% loop over the possible regularization values
        parfor i = 1:length(lvect)
            %%%% train the model
            outputs = py.logisticRegression.logisticRegression(XTrain(:).', YTrain(:).', l(i, :)*scale);
            Beta =  double(py.array.array('d',py.numpy.nditer(outputs(1))));
            Bias = double(py.array.array('d',py.numpy.nditer(outputs(2))));
            lps = double(py.array.array('d',py.numpy.nditer(outputs(3))));
            
            %%%% measure testing error
            YpTest = predictLogisticRegression(XTest, Beta, Bias, lps);
            perClassErrorTest(s, i) = sum(abs(YTest(:) - YpTest(:)))/length(YTest)*100;
            %%%% measure training error
            YpTrain = predictLogisticRegression(XTrain, Beta, Bias, lps);
            perClassErrorTrain(s, i) = sum(abs(YTrain(:) - YpTrain(:)))/length(YTrain)*100;

        end
    end
    
    %%%% measure the mean and standard error of the mean of training and testing errors 
    meanPerClassErrorTrain = mean(perClassErrorTrain);
    semPerClassErrorTrain = std(perClassErrorTrain)/sqrt(numSamples);
    
    meanPerClassErrorTest = mean(perClassErrorTest);
    semPerClassErrorTest = std(perClassErrorTest)/sqrt(numSamples);
    [~, ix] = min(meanPerClassErrorTest);
    ibest = find(meanPerClassErrorTest <= (meanPerClassErrorTest(ix)+semPerClassErrorTest(ix)), 1, 'last');
    lbest = l(ibest, :); % best regularization term based on minError+SE criteria
    
    %%%%%% plot coss-validation results
    figure
    hold on
    errorbar(lvect, meanPerClassErrorTrain, semPerClassErrorTrain, 'b')
    errorbar(lvect, meanPerClassErrorTest, semPerClassErrorTest, 'r')
    plot(lvect(ibest), meanPerClassErrorTest(ibest), 'ro')
    xlim([lvect(2) lvect(end)])
    set(gca, 'xscale', 'log')
    xlabel('regularization parameter')
    ylabel('classification error (%)')
    hold off
    legend('training error', 'test error', 'best parameter')
else
    lbest = [0. 0.]; % no-cross validation (set model to no regularization)
end

    %%%%% train the best model using all data
    outputs = py.logisticRegression.logisticRegression(X(:).', Y(:).', lbest(:).'*scale);
    %%%%% save results
    Summary.Model.Beta =  double(py.array.array('d',py.numpy.nditer(outputs(1))));
    Summary.Model.Bias = double(py.array.array('d',py.numpy.nditer(outputs(2))));
    Summary.Model.lps = double(py.array.array('d',py.numpy.nditer(outputs(3))));
    Summary.Model.perClassEr = double(py.array.array('d',py.numpy.nditer(outputs(4))));
    Summary.Model.cost = double(py.array.array('d',py.numpy.nditer(outputs(5))));
    Summary.Model.l2_regularization = lbest(1)*scale;
    Summary.Model.l1_regularization = lbest(2)*scale;
    cost_i =  double(py.array.array('d',py.numpy.nditer(outputs{6}.cost_per_iter)));
    perClassEr_i =  double(py.array.array('d',py.numpy.nditer(outputs{6}.perClassEr_per_iter)));

%%%%% plot optimization results 
figure
subplot(211)
plot(cost_i)
xlabel('iteration')
ylabel('cost')
subplot(212)
plot(perClassEr_i)
xlabel('iteration')
ylabel('classification error (%)')
ylim([0 50])
end


