%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% <subspaces>
% Copyright (C) 2016 Gamaleldin F. Elsayed and John P. Cunningham (see full notice in README)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Ix] = sampleRandSubspaces(dim, useCov, IndexFn, numSamples)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function evaluates the alignment index between two datasets
% occupying two subspaces.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
%       - dim: dimensionality of the subspace.
%       - useCov: sample subspaces based on some data covariance. This
%       ensures that the subspaces are sampled from the same data space.
%       This data alignment option can be disabled by setting useCove to
%       identity matrix.
%       - IndexFn: contains function name that calculates an index between
%       two samples spaces.
%       - numSamples: number of samples. 
%       - varargin: other optional inputs. These can be used for other
%       inputs for the IndexFn.
% Outputs:
%       - Ix: this is the result distribution of indices based on random
%       subspaces.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Ix] = sampleRandSubspaces(dim, useCov, IndexFn, numSamples, varargin)
%% Project Neural Activity on random directions for significance analysis 
%%% generate random unit basis (random unit vectors in neural space)

rng shuffle
N = size(useCov, 1);                                                       % space size
[U, S, ~] = svd(useCov);                                                   
biasMtx = U*diag(sqrt(diag(S)));                                           % evaulates the biasing matrix that bias the random vectors to data space
Ix = nan(numSamples,1);
parfor s = 1:numSamples
    %% sample random subspace 1
    w1 = randn(N, dim);                                                    % initially sample uniformly
    [~, w1] = normVects(w1); 
    w1 = orth(biasMtx*w1);                                                 % bias the sampled vectors to data space
    %% sample random subspace 2
    w2 = randn(N, dim);
    [~, w2] = normVects(w2); 
    w2 = orth(biasMtx*w2);
    %% evaluate statistic measure between space 1 and 2
    if isempty(varargin)
        Ix(s) = feval(IndexFn, w1, w2);
    else
        Ix(s) = feval(IndexFn, w1, w2, varargin);
    end
end
end

