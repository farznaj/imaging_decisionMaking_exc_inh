%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% <subspaces>
% Copyright (C) 2016 Gamaleldin F. Elsayed and John P. Cunningham (see full notice in README)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ix = alignIx(w1, w2, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function evaluates the alignment index between two datasets
% occupying two subspaces.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
%       - D: is the matrix of size (neurons x observations) that contains 
%         the responses. 
%       - w: are the orthonormal basis of size (neurons x dimensionality2) 
%         that defines the subspace that we want to test if response 1 is
%         aligned to. 
% Outputs:
%       - Ix: is the resulting alignment index of response1 to subspace 2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Ix = alignIx(D, w)
    w = orth(w);                                                           % make sure the basis are orthogonal
    dim = size(w, 2);                                                      % dimensionality of the subspace
    C = (D*D');                                                            % calculate covariance (note, scale of covariance is not important as it cancels out in the alignment index formula)
    s = svd(C);                                                            % singular values of the covariance
    Ix = trace(w'*C*w)./sum(s(1:dim));                                     % calculate the alignment index
end