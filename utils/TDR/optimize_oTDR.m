function [Q, g] = optimize_oTDR(dataTensor, codedParams, trCountmtx, spcConstraint)
% It takes all the times and tries to find the best axes.
% Q: neurons x numberTunedFeatures : beta coefficients for all predictors except the offset.

[T, N, C] = size(dataTensor);
if isempty(spcConstraint)
    projSpc = eye(N);
else
    spcConstraint = orth(spcConstraint); % make sure constraint space is orthonormal;
    projSpc = spcConstraint*spcConstraint'; % projection matrix on the space
end

if isempty(trCountmtx)
    trCountmtx = ones(N, C);
end

numSignals = size(codedParams, 2);
Q = orth(randn(N, numSignals-1));
Beta0 = randn(N, T);
g = ones(T, numSignals-1);

%% random initialization within data space
maxIter = 200;
stp = 10000;
for i = 1:maxIter
    i
    for t = randperm(T)
        Betas = [Q*diag(g(t,:)) Beta0(:,t)];
        [f(i,t), gradf, ~,~] = TDRobjectiveFn(dataTensor(t, :, :), codedParams, permute(Betas, [3 1 2]), trCountmtx);
        gradf = squeeze(gradf);
        %         Betas(:,j) = Betas(:,j)-stp*gradf(:,j); % update gradient with respect to data R{j}
        Betas(:,1:numSignals-1) = Betas(:,1:numSignals-1)-stp*gradf(:,1:numSignals-1); % update gradient with respect to all data
        Betas(:,numSignals) = Betas(:,numSignals)-stp*gradf(:,numSignals);
        Q = Betas(:, 1:numSignals-1); % neurons x numTunedFeatures
        Beta0(:,t) = Betas(:, numSignals);
        [u,~,v] = svd(projSpc*Q, 0); %u: neurons x numTunedFeatures ; v: numTunedFeatures x numTunedFeatures
        g(t,:) = diag((Q')*(u*v')); % diag((Q')*(u*v')) is of size numTunedFeatures x numTunedFeatures
        Q = (u*v'); % u*v’ is the nearest orthonormal matrix.... the closest projection to the space of orthonormal matrices
    end
    %%
    
end
figure;
plot(mean(f,2))

end











