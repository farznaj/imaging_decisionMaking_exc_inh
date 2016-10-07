function [dRAs, normdRAs, Summary] = runTDR(dataTensor, numPCs, codedParams, trCountmtx, kfold)
% dRAs:  times x neurons x predictors; beta coefficients normalized by normdRAs
% normdRAs: times x predictors; 2-norm of columns of reshaped beta (neurons x (times x predictors))
% Summary: includes for each time point: error, R2 of the model, R2 of each
% predictor, and regularization paramter lambda (for each neuron)

[T, N, C] = size(dataTensor);
if isempty(trCountmtx);
    trCountmtx = ones(N, C); % neurons x trials
end
%% denoise step
if numPCs < N
    XN = reshape(permute(dataTensor,[1 3 2]), [], N); % (frames x trials) x neurons
    PCs = pca(XN); % neurons x neurons
    XN = bsxfun(@minus, XN, mean(XN))*(PCs(:, 1:numPCs)*PCs(:, 1:numPCs)');
    dataTensor = permute(reshape(XN', N, T, C), [2 1 3]);
end

%% Solve a regression problem for each neuron at each time point  
K = size(codedParams, 2);
sqrtTrCountmtx = sqrt(trCountmtx);
Betas = nan(T, N, K); % beta coefficients % times x neurons x (number of features)
lambda = nan(T, N); % regularization term % times x neurons
for t = 1:T
   parfor n = 1:N
       scaledCodedParams = repmat(sqrtTrCountmtx(n, :)', 1, K).*codedParams; % scaled coded parameters by the square root of trial counts % trials x 3, stimRate, choice and ones after mean subtraction and l-2 normalization.
       scaledRbar_tn = sqrtTrCountmtx(n, :)'.*reshape(dataTensor(t, n, :), C, 1); % scaled dataTensor by the square root of trial counts % trials x 1, response of neuron n at time t (responses are already downsampled, mean subtracted, feature normalized, and denoised)
       [Betas(t, n, :), lambda(t, n)] = linearRegression(scaledRbar_tn, scaledCodedParams, C, kfold);
   end
end
[Summary.f_t, ~, Summary.R2_t, Summary.R2_tk] = TDRobjectiveFn(dataTensor, codedParams, Betas, trCountmtx);
Summary.lambda = lambda; % times x neurons

%% divide by magnitude and direction  
[normdRAs, dRAs] = normVects(reshape(permute(Betas, [2, 1, 3]), N, T*K)); % reshape beta to neurons x (times x predictors) and then normalize each column
dRAs(:, normdRAs<=eps) = 0; %set the direction to zero if the magnitude of the vector is zero
dRAs = permute(reshape(dRAs, N, T, K), [2 1 3]); % times x neurons x predictors
normdRAs = reshape(normdRAs, T, K); % times x predictors
end


function [B, lambda] = linearRegression(R, P, numConds, kfold)
lambdas =  [0 2.^[0:12]*0.01 inf];
T = size(R,1)/numConds;
K = size(P, 2);

if islogical(kfold) || isempty(kfold)
    if kfold
        R = reshape(R, T, numConds);
        P = reshape(P, T, numConds, K);
        erTrain = nan(length(lambdas), numConds);
        erTest = nan(length(lambdas), numConds);
        parfor c = 1:numConds
        msk = false(numConds, 1);
        msk(c) = true;
        RTest = R(:, msk);
        PTest = reshape(P(:, msk, :), T, K);

        RTrain =  R(:, ~msk).';
        PTrain = reshape(P(:, ~msk, :), T*(numConds-1), K);
        [erTrain(:, c), erTest(:, c)] =  ridgeRegress(PTrain,RTrain(:), PTest, RTest, lambdas);
        end
        [minEr,ix] = min(bsxfun(@times, sum(erTest,2), 1));
        lambda = lambdas(ix);

        R =  R(:, :).';
        P = reshape(P(:, :, :), T*(numConds), K);

        [Er,~, B] =  ridgeRegress(P, R(:), P, R(:), lambda);
    else
        lambda = nan;
        B = regress(R, P);
    end
else
    R = reshape(R, T, numConds);
    P = reshape(P, T, numConds, K);
    numSamples = 10;
    erTrain = nan(length(lambdas), numSamples);
    erTest = nan(length(lambdas), numSamples);
    for c = 1:numSamples
        shfl = randperm(numConds);
        Rs = R(:, shfl);
        Ps = P(:, shfl, :); 
        mskTrain = 1:floor((kfold-1)/kfold*numConds);
        mskTest =  floor((kfold-1)/kfold*numConds)+1:numConds;       
        RTest = Rs(:, mskTest).';
        PTest = reshape(Ps(:, mskTest, :), [], K);

        RTrain =  R(:, mskTrain).';
        PTrain = reshape(P(:, mskTrain, :), [], K);
        [erTrain(:, c), erTest(:, c)] =  ridgeRegress(PTrain, RTrain(:), PTest, RTest, lambdas);
    end
    [minEr,ix] = min(bsxfun(@times, mean(erTest,2), 1));
    lambda = lambdas(ix);

    R =  R(:, :).';
    P = reshape(P(:, :, :), T*(numConds), K);

    [Er,~, B] =  ridgeRegress(P, R(:), P, R(:), lambda);
    
    
end
end
function [erTrain,erTest,Xlast]=  ridgeRegress(Atrain, Btrain, Atest, Btest, lambdas) % linear regression with l2 penalty
[~,n]=size(Atrain);

erTrain = nan(length(lambdas),1);
erTest = nan(length(lambdas),1);
for i=1:length(lambdas)
    lambda = (lambdas(i)*sqrt(mean(Atrain(:).^2)))^2;
    if isinf(lambda)
        X = zeros(size(Atrain,2), size(Btrain,2));
    else
        X=(Atrain'*Atrain+diag(lambda*ones(n,1)))\(Atrain'*Btrain);
        %     X=(Atrain'*Atrain+diag([lambda*ones(n-1,1); 0]))\(Atrain'*Btrain);
    end
    erTrain(i)=norm(Atrain*X-Btrain,'fro')^2;%./norm(Btrain,'fro')^2; 
    erTest(i)=norm(Atest*X-Btest,'fro')^2;%./norm(Btest,'fro')^2; 
end
Xlast = X;

end



