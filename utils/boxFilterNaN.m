function filtered = boxFilterNaN(data, nPts, dim, causal)
% This function is exactly like matt's boxFilter, except I am using
% nancumsum to take care of traces with nans.
%
% filtered = boxFilter(data, filterWidth [, dim] [, causal])
%
% Implement a fast boxcar filter.
%
% filtered is the same size as data.
% filterWidth is the number of points of the boxcar.
% dim optionally specifies the dimension to operate on (default 1).
% causal optionally specifies whether to use a causal filter (if 1) or a
%   window centered around each point (if 0). If using causal=0 for a
%   centered window, and filterWidth is even, the filter will have one more
%   causal point than acausal points. Default 1.
%
% The initial points are handled by replicating the "front" of the matrix.
% (The same is true for the end if using causal=0.) Thus, the first points
% along dimension dim in filtered are the same as in data.


%% Optional arguments

if ~exist('dim', 'var')
  dim = 1;
end

if ~exist('causal', 'var')
  causal = 1;
end


%% Manipulate array to bring dimension to filter to index 1

% We'll swap the dimension of interest with the first dimension, operate on
% dimension 1, and swap back later.
if dim > 1
  dims = 1:ndims(data);
  dims(1) = dim;
  dims(dim) = 1;
  
  data = permute(data, dims);
end

siz = size(data);


%% Add buffer to front or front and end of data

% First, capture the first row/page/whatever
data = reshape(data, siz(1), []);
data1 = data(1, :);
dataEnd = data(end, :);


if causal
  data = [repmat(data1, [nPts 1]); data];
else
  nPts2 = ceil(nPts / 2);
  data = [repmat(data1, [nPts2 1]); data; repmat(dataEnd, [nPts - nPts2 1])];
end


%% Use cumsum to filter. Trim off buffers in same step

% cs = cumsum(data, 1);
cs = nancumsum(data, 1, 2);
filtered = cs(nPts+1:end, :) - cs(1:end-nPts, :);


%% Normalize to get mean, since we've only added

filtered = filtered / nPts;


%% Restore the filtered data to its original shape (after swapping dimensions)

filtered = reshape(filtered, siz);


%% If we permuted the dimensions earlier, restore original ordering

if dim > 1
  filtered = permute(filtered, dims);
end
