function new = insertElement(arr, inds, vals)
% Adapted from Matt's code insertIntoArray. inds refers to row indeces. Accepts matrices too.
%
% new = insertIntoArray(arr, inds, val)
% new = insertIntoArray(arr, inds, vals)
%
% Construct a new 1-D array that looks just like the old 1-D array provided
% in arr, but has values vals "inserted" at indices inds.
%
% Indices should be relative to the original array. This means that if
% length(inds) > 1, the new values will not all end up at the indices in
% inds (see example).
%
% If vals is the same length as inds, corresponding elements of vals will
% be used. If val is of length 1, the same value will inserted for each
% element of inds. If the length of vals is not 1, it must be the same
% length as inds.
%
% inds should not contain values greater than length(arr)+1, and inds
% should not contain repeated values.
%
% The original array orientation is preserved.
%
% Examples:
% insertIntoArray(1:5, [2 6], [10 11])
% ans =
%      1    10     2     3     4     5    11
%
% insertIntoArray((1:4)', [1 5], NaN)
% ans =
%    NaN
%      1
%      2
%      3
%      4
%    NaN


% Figure out new size (need to infer dimension to expand)
siz = size(arr);
% dim = find(siz > 1);
% siz(dim) = max([siz(dim) + length(inds), max(inds) + length(inds) - 1]);
siz(1) = size(arr,1) + length(inds);

% Fix indices to deal with accumulating offsets
inds = inds + (0:length(inds) - 1);
logInds = false(siz);
logInds(inds,:) = true;

% Construct new array
new = zeros(siz);
new(logInds) = vals;
new(~logInds) = arr;


