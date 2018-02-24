% set upper triangular of a matrix to nan

function A = triu_nan(A)

A(triu(true(size(A)))) = nan;
