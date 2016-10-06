[PCs, s, l] = pca(X);

% X: trials x neurons --> 200   400

% PCs: neurons x PCs --> 400   135
% s: trials x PCs --> 200   135
% l: PCs x 1 --> 135 x 1
   
% PCs: (neurons x PCs) coeff of neurons in the PC space.
% s: (trials x PCs) projection of X onto the PC space. (s = bsxfun(@minus, X, mean(X))*PCs;)
% l: (PCs x 1) principal component variances, ie eigenvalues of the covariance matrix of X.

a = bsxfun(@minus, X, mean(X))*PCs; % this is same as s: projection of X onto the PC space.
