function [Q] = optimize_oTDR(dataTensor, codedParams, trCountmtx, spcConstraint)
[T, N, C] = size(dataTensor);
if isempty(spcConstraint)
projSpc = eye(N);
else
    spcConstraint = orth(spcConstraint); % make sure contraint space is orthonormal;
    projSpc = spcConstraint*spcConstraint'; % projection matrix on the space
end

if isempty(trCountmtx)
    trCountmtx = ones(N, C);
end

numSignals = size(codedParams, 2);
Q = orth(randn(N, numSignals-1));
Beta0 = randn(N, T);
a = ones(T, numSignals-1);
%% random initialization within data space
maxIter = 200;
stp = 10000;
for i = 1:maxIter
 i
for t = randperm(T)
        Betas = [Q*diag(a(t,:)) Beta0(:,t)];
        [f(i,t), gradf, ~,~] = TDRobjectiveFn(dataTensor(t, :, :), codedParams, permute(Betas, [3 1 2]), trCountmtx);
        gradf = squeeze(gradf);
%         Betas(:,j) = Betas(:,j)-stp*gradf(:,j); % update gradient with respect to data R{j}
        Betas(:,1:numSignals-1) = Betas(:,1:numSignals-1)-stp*gradf(:,1:numSignals-1); % update gradient with respect to all data
        Betas(:,numSignals) = Betas(:,numSignals)-stp*gradf(:,numSignals);
        Q = Betas(:, 1:numSignals-1);
        Beta0(:,t) = Betas(:, numSignals);
        [u,~,v] = svd(projSpc*Q, 0);
        a(t,:) = diag((Q')*(u*v'));
        Q = (u*v');
end
%%   
    
end
figure;
plot(mean(f,2))
end











