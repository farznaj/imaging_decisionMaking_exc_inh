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
%%
scale = sqrt(mean(X(:).^2));

if kfold>0
    lvect = [0; 10.^(-5:0.5:1)'];
    if strcmpi(regType, 'l2')
        l = [lvect zeros(length(lvect), 1)];
    elseif strcmpi(regType, 'l1')
        l = [zeros(length(lvect), 1) lvect];
    end
    numSamples = 100;
    L = length(Y);

    perClassErrorTest = nan(numSamples, length(lvect));
    perClassErrorTrain = nan(numSamples, length(lvect));
    for s = 1:numSamples
        shfl = randperm(L);
        Ys = Y(shfl);
        Xs = X(shfl, :); 
        
        YTrain = Ys(1:floor((kfold-1)/kfold*L));
        YTest = Ys(floor((kfold-1)/kfold*L)+1:L);
        
        XTrain = Xs(1:floor((kfold-1)/kfold*L), :);
        XTest = Xs(floor((kfold-1)/kfold*L)+1:L, :);
        parfor i = 1:length(lvect)
            outputs = py.logisticRegression.logisticRegression(XTrain(:).', YTrain(:).', l(i, :)*scale);
            Beta =  double(py.array.array('d',py.numpy.nditer(outputs(1))));
            Bias = double(py.array.array('d',py.numpy.nditer(outputs(2))));
            lps = double(py.array.array('d',py.numpy.nditer(outputs(3))));
            YpTest = predictLogisticRegression(XTest, Beta, Bias, lps);
            perClassErrorTest(s, i) = sum(abs(YTest(:) - YpTest(:)))/length(YTest)*100;

            YpTrain = predictLogisticRegression(XTrain, Beta, Bias, lps);
            perClassErrorTrain(s, i) = sum(abs(YTrain(:) - YpTrain(:)))/length(YTrain)*100;

        end
    end
    
    meanPerClassErrorTrain = mean(perClassErrorTrain);
    semPerClassErrorTrain = std(perClassErrorTrain)/sqrt(numSamples);
    
    meanPerClassErrorTest = mean(perClassErrorTest);
    semPerClassErrorTest = std(perClassErrorTest)/sqrt(numSamples);
    [~, ix] = min(meanPerClassErrorTest);
    ibest = find(meanPerClassErrorTest <= (meanPerClassErrorTest(ix)+semPerClassErrorTest(ix)), 1, 'last');
    lbest = l(ibest, :);
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
    lbest = [0. 0.];
end

    outputs = py.logisticRegression.logisticRegression(X(:).', Y(:).', lbest(:).'*scale);
    %%
    Summary.Model.Beta =  double(py.array.array('d',py.numpy.nditer(outputs(1))));
    Summary.Model.Bias = double(py.array.array('d',py.numpy.nditer(outputs(2))));
    Summary.Model.lps = double(py.array.array('d',py.numpy.nditer(outputs(3))));
    Summary.Model.perClassEr = double(py.array.array('d',py.numpy.nditer(outputs(4))));
    Summary.Model.cost = double(py.array.array('d',py.numpy.nditer(outputs(5))));
    Summary.Model.l2_regularization = lbest(1)*scale;
    Summary.Model.l1_regularization = lbest(2)*scale;
    cost_i =  double(py.array.array('d',py.numpy.nditer(outputs{6}.cost_per_iter)));
    perClassEr_i =  double(py.array.array('d',py.numpy.nditer(outputs{6}.perClassEr_per_iter)));

%% 
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


