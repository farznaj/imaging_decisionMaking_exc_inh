function ACOMs = fastCOMsA(A, siz)
% ACOMs = fastCOMsA(A, siz)
% 
% Compute the center of mass (COM) of each neuron in A. siz should be the
% size of the un-reshaped image (e.g., size(medImage{2})). ACOMs uses the
% standard image convention for dimension order, so rows are [y x].


% The COM is the first moment of the data: sum(x*w), where w is the
% weight of that point. So we can get this very efficiently.

% Produce "x" in the equation above for each point in the image
[X, Y] = meshgrid(1:siz(2), 1:siz(1));

% Make the columns of A sum to 1, so that the values are proper weights
A = bsxfun(@rdivide, A, sum(A));

% Take the first moment
ACOMs = [A' * Y(:), A' * X(:)];
