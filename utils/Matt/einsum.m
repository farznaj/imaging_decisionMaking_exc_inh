function C = einsum(A, B, indsA, indsB)
% C = einsum(A, B, indsA, indsB)
%
% This function implements a simpler version of einsum for Python. It does
% inner products of n-dimensional matrices.
%
% A and B should be n-dimensional matrices. They do not need to have the
% same number of indices.
%
% indsA and indsB should be the indices to collapse when performing the
% inner product. That is, these are the indices that will go away in the
% result. More than one index may be used. These lists must be the same
% length, and the corresponding dimensions must have the same numbers of
% elements. These lists are 1-indexed in keeping with Matlab convention.
%
% Example: if A is M x N x P x Q, and B is Q x P x J, calling:
%   C = einsum(A, B, [4 3], [1 2])
%   ... will produce a result of size M x N x J.
%
% Note: unlike Python's einsum, indices may not be repeated.
%
% Copyright Matt Kaufman 2013.


%% Error checking

% Same number of indices to collapse for A and B
if length(indsA) ~= length(indsB)
  error('einsum:nIndexMismatch', ...
    'Number of indices to sum over must match');
end

% Ensure no dimension is repeated
if length(unique(indsA)) ~= length(indsA)
  error('einsum:repeatedIndex', ...
    'Cannot repeat an index');
end

% Ensure indices are bounded appropriately
if max(indsA) > ndims(A) || max(indsB) > ndims(B)
  error('einsum:indexTooBig', ...
    'Indices must be no greater than the number of dimensions in the matrix');
end

% For each corresponding dimension to take inner product, ensure they're
% the same length
sizA = size(A);
sizB = size(B);
if any(sizA(indsA) ~= sizB(indsB))
  error('einsum:dimSizeMismatch', ...
    'Dimensions on which to collapse for A and B must have same size');
end


%% Permute A and B so that the first nInds indices are the ones to operate on

nInds = length(indsA);

% Make permutation vector
dimsA = 1:ndims(A);
dimsB = 1:ndims(B);

dimsA(indsA) = [];
dimsB(indsB) = [];

dimsA = [indsA dimsA];
dimsB = [indsB dimsB];

% Permute
A = permute(A, dimsA);
B = permute(B, dimsB);


%% Reshape for matrix multiplication

sizA = size(A);
sizB = size(B);

A = reshape(A, prod(sizA(1:nInds)), prod(sizA(nInds+1:end)));
B = reshape(B, prod(sizB(1:nInds)), prod(sizB(nInds+1:end)));


%% Multiply

C = A' * B;


%% Reshape result

C = squeeze(reshape(C, [sizA(nInds+1:end) sizB(nInds+1:end)]));
