function [PCs_t, Summary] = pca_t(dataTensor, numPCs)
[T, N, ~] = size(dataTensor);
PCs_t = nan(T, N, numPCs);
var_t = nan(T, numPCs);
totalVar_t = nan(T, 1);
for t = 1:T
    dataTensor_t = squeeze(dataTensor(t, :, :));
    
    [~, s, v] = svd(bsxfun(@minus, dataTensor_t.', mean(dataTensor_t.')), 'econ');
    s = diag(s);
    PCs_t(t, :, :) = v(:, 1:numPCs);
    var_t(t, :) = s(1:numPCs);
    totalVar_t(t) = sum(s); 
    
end
Summary.totalVar_t = totalVar_t;
Summary.var_t = var_t;
end
