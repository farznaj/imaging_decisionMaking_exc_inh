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
% l1 regularization parameter is obtain by kfold cross validation with 1SE
% criteria. This function interfaces and calls a python function 
% (lassoLinearSVM.py), which does all the heavy-duty work.
%
% Inputs:
%   X: features matrix of size (trials x neurons).
%   Y: labels vector of size (trials x 1).
%   kfold: cross-validation folds.
% Outputs:
%   SVMl1Model: structure with the following fields
%   	.Model: the scalar slope of the regression model
%           .Beta: linear svm coefficients of size (neuorns x 1).
%           .Bias: bias component.
%           .regularization: l1-regularization parameter used.
%           .CVerror: Cross validation error of the model
%           .Object: svm object passed from the python function
%   	.Standard: values used in feature normalization and scaling
%           .meanX: center features to have zero mean.
%           .stdX: scale features to have unity std.
%   	.CrossValidation: field that contains information the cross
%   	validation proceudure used to find the regularization parameters.
%           .kfold: kfold used in cross validation.
%           .regularization: regularization valused examined.
%           .meanCVerror: mean cross validation error from 100 runs. Different entries reflect result from different values of regularization specified above.
%           .semCVerror: standard error of the mean of cross validation
%           error from 100 runs. Different entries reflect result from different values of regularization specified above.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [SVMl1Model] = lassoLinearSVM(X, Y, kfold)

% add path to python. You need to be in the directory of lassoLinearSVM
% functions.
if count(py.sys.path,'') == 0
insert(py.sys.path,int32(0),'');
end

outputs = py.lassoLinearSVM.lassoLinearSVM(X(:).', Y(:).', kfold);
w = cell2mat(cell(py.list(outputs{1})));
b = cell2mat(cell(py.list(outputs{2})));
bestCVerror = outputs{3};
bestc = outputs{4};
linear_svm = outputs{5};
meanCVerror = cell2mat(cell(py.list(outputs{6})));
semCVerror = cell2mat(cell(py.list(outputs{7})));
c = cell2mat(cell(py.list(outputs{8})));
meanX = cell2mat(cell(py.list(outputs{9})));
stdX = cell2mat(cell(py.list(outputs{10})));

figure
subplot(211)
hold on
errorbar(c, meanCVerror,  semCVerror, 'k')
plot(bestc, bestCVerror, 'ro')
hold off
ylim([0 1])
set(gca, 'xscale', 'log')
xlabel('inverse of lasso-regularization parameter (c)')    
ylabel('cross-validation error')    
subplot(212)
plot(sort(abs(w), 'descend'))
xlabel('ordered neurons')
ylabel('|w|')

SVMl1Model.Model.Beta = w;
SVMl1Model.Model.Bias = b;
SVMl1Model.Model.regularization = bestc;
SVMl1Model.Model.CVerror = bestCVerror;
SVMl1Model.Model.Object = linear_svm;
SVMl1Model.Standard.meanX = meanX;
SVMl1Model.Standard.stdX = stdX;
SVMl1Model.CrossValidation.kfold = kfold;
SVMl1Model.CrossValidation.regularization = c;
SVMl1Model.CrossValidation.meanCVerror = meanCVerror;
SVMl1Model.CrossValidation.semCVerror = semCVerror;

end