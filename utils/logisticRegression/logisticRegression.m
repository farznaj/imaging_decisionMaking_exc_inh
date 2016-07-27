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
%   l: first element; l2-regularization parameter and second element l1-regularization parameter.
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
function [Summary] = logisticRegression(X, Y, l)
% add path to python. You need to be in the directory of lassoLinearSVM
% functions.
if count(py.sys.path,'') == 0
insert(py.sys.path,int32(0),'');
end
%%
outputs = py.logisticRegression.logisticRegression(X(:).', Y(:).', l(:).');
Summary.Model.Beta =  double(py.array.array('d',py.numpy.nditer(outputs(1))));
Summary.Model.Bias = double(py.array.array('d',py.numpy.nditer(outputs(2))));
Summary.Model.lps = double(py.array.array('d',py.numpy.nditer(outputs(3))));
Summary.Model.perClassEr = double(py.array.array('d',py.numpy.nditer(outputs(4))));
Summary.Model.cost = double(py.array.array('d',py.numpy.nditer(outputs(5))));
Summary.Model.l2_regularization = l(1);
Summary.Model.l1_regularization = l(2);
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