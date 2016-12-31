%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Copyright (C) 2016 Gamaleldin F. Elsayed and Farzaneh Najafi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gamaleldin F. Elsayed
% 
% regressCommonSlopeModel.m
%
% This function solves the model yi = a xi + bi for the variables a and 
% bi for i \in {1, ...., N}. Unlike the 
%
% Inputs:
%   Xs: cell of size (N x 1) where each element is a vector of length 
%    (frames x 1)
%   Ys: cell of size (N x 1) where each element is a vector of length 
%    (frames x 1)
% Outputs:
%   a: the scalar slope of the regression model
%   bs: vector of size (N x 1), which includes the offset elements
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [a, bs] = regressCommonSlopeModel(Xs, Ys, maxIter)
if ~exist('maxIter', 'var')
    maxIter = 100;                              % maximum number of iterations allowed.
end
N = length(Xs);                             % number of variables (i.e., # neurons)
a = randn;                                  % initial value for a
bs = randn(N, 1);                           % initial value for bs
fij = nan(maxIter, N);                      % cost function
normTerm = norm(vertcat(Ys{:}))^2;          % normalize objective by the norm of Ys
for i = 1:maxIter
    Ybar = cell(N, 1);                      % initialize Ybar, these are the Ys with subtracted offset
    parfor j = 1:N
        Ybar{j} = Ys{j}- bs(j);             % calculate Ybar
        bs(j) = mean(Ys{j}-a*Xs{j});        % calculate the solution of bs given a fixed a
    end    
    x = vertcat(Xs{:});                     % vectorize all the Xs to be used for a calculation
    a = x'*vertcat(Ybar{:})/(x(:)'*x(:));   % solution for one slope a given fixed bs
    [fij(i, :)] = objFnRegressCommonSlopeModel(Xs, Ys, a, bs);  % calculate objective value
    fij(i, :) = fij(i, :)./normTerm;            % normalize objective by the total norm of Ys

    % if no change in objective value for 2 iterations declare convergence
    if i>2
        if abs(sum(fij(i, :), 2)-sum(fij(i-1, :), 2))<sqrt(eps) && abs(sum(fij(i, :), 2)-sum(fij(i-2, :), 2))<sqrt(eps)
            break; 
        end
    end
end

% plot summary figure
figure;
plot(sum(fij,2));
xlabel('iter')
ylabel('cost value')
end