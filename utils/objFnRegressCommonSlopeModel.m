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
% This function calculates the cost ||yi - (a xi - bi)|| and the current
% solution of a and bi for i \in {1, ...., N}. Unlike the 
%
% Inputs:
%   Xs: cell of size (N x 1) with each element is a vector of length 
%    (frames x 1)
%   Ys: cell of size (N x 1) with each element is a vector of length 
%    (frames x 1)
%   ain: the scalar slope of the regression model
%   bin: vector of size (N x 1), which includes the offset elements
% Outputs:
%   f: vector of cost function for each of the N elements
%   aout: the scalar slope of the regression model
%   bout: vector of size (N x 1), which includes the offset elements
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f, aout, bout] = objFnRegressCommonSlopeModel(Xs, Ys, ain, bin)
N = length(Xs);
Ybar = cell(N, 1);
f = nan(N,1);
bout = nan(N,1);
aout = nan;
parfor j = 1:N
    bout(j) = mean(Ys{j}-ain*Xs{j});
    Ybar{j} = Ys{j}- bin(j);
    f(j) = norm(Ys{j} - ain*Xs{j} - bin(j), 'fro')^2;
end    
x = vertcat(Xs{:});
aout = x'*vertcat(Ybar{:})/(x(:)'*x(:));
end