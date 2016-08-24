function dataTensor_proj = projectTensor(dataTensor, w)
[~, w] = normVects(w);
[T, N, R] = size(dataTensor);
K = size(w,2);
XN = reshape(permute(dataTensor, [1 3 2]), T*R, N);

XN_proj = bsxfun(@minus, XN, mean(XN))*w;

dataTensor_proj = permute(reshape(XN_proj, T, R, K), [1 3 2]);
end