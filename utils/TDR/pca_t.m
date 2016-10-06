function [PCs_t, Summary] = pca_t(dataTensor, numPCs)
[T, N, ~] = size(dataTensor);
PCs_t = nan(T, N, numPCs);
var_t = nan(T, numPCs);
totalVar_t = nan(T, 1);
for t = 1:T
    dataTensor_t = squeeze(dataTensor(t, :, :)); % neurons x trials at time t
    
    [~, s, v] = svd(bsxfun(@minus, dataTensor_t.', mean(dataTensor_t.')), 'econ'); % perform svd on trials x neurons after mean subtraction (mean across trials for each neuron)
    % s: trials x trials (trials represent the number of singular vectors)
    % v: neurons x trials (columns: singular vectors)
    s = diag(s); % singular values corresponding to each column of v.
    PCs_t(t, :, :) = v(:, 1:numPCs); % (neurons x PCs): PCs for time t. only take numPCs columns of v.
    var_t(t, :) = s(1:numPCs); % eigenvalues of each PC (variance explained by each PC) for time t.
    totalVar_t(t) = sum(s); % sum of all eigenvalues (total variance) for time t.
    
end
Summary.totalVar_t = totalVar_t; % times x 1 : total variance explained (by all singular vectors not just numPCs) at each time point.
Summary.var_t = var_t; % times x numPCs : variance explained by each PC at each time point.
end
