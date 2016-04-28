% Remember: you can use the script svmUnderstandIt to understand how some
% of the matlab functions related to SVM classification work.
windowAvgFlg = true;
pcaFlg = true;
stMs = round(500/frameLength);
enMs = round(900/frameLength);

thAct = 1e-3; % could be a good th for excluding neurons w too little activity.
numRand = 1; % 50; 100; % you tried values between [50 500], at nrand=500 mismatches (computed on the b averaged across all iters) are actually worse compared to the average mismatch computed from single runs.  nrand=200 seems to be a good choice.
doplots = true;


method = 'svm'; % 'logisticRegress';  % classification method for neural population analysis.
rng(0, 'twister'); % Set random number generation for reproducibility


%% Set Y: the response vector

choiceVec0 = allResp_HR_LR';  % trials x 1;  1 for HR choice, 0 for LR choice.
choiceVec0(outcomes~=1) = NaN; % analyze only correct trials.
fprintf('N trials for LR and HR = %d  %d\n', [sum(choiceVec0==0), sum(choiceVec0==1)])

%% Set X: the predictor matrix (trials x neurons) that shows average of spikes for a particular epoch for each trial and neuron.

% Set the epoch of neural responses that you want to analyze.
minStimFrs = floor(min(stimdur*1000)/frameLength); % minimum stimdur of all trials in frames
nPostFrs = size(traces_al_sm,1) - (eventI_stimOn-1); % number of frames after the stimOn (including stimOn frame) in the aligned traces.
epLen = min(nPostFrs, minStimFrs); % length of the epoch we are going to anayze.

epStart = eventI_stimOn + stMs; %20; round(500/frameLength); % eventI_stimOn; % start of the epoch
epEnd = min(eventI_stimOn + enMs, eventI_stimOn+epLen-1); % eventI_stimOn+epLen-1; % end of the epoch
ep = epStart : epEnd; % frames in traces_al_sm that will be used for analysis. % for now lets use spike counts over the entire stim frames.

% Compute average of spikes per frame during epoch ep.
spikeAveEp0 = squeeze(nanmean(traces_al_sm(ep,:,:)))'; % trials x units.


%% Remove (from X) neurons with average activity during epoch ep in too few trials. or with too little average activity during epoch ep and averaged across all trials.
%
thMinFractTrs = .05; %.01; % a neuron must be active in >= .1 fraction of trials to be used in the population analysis.
thTrsWithSpike = 3; ceil(thMinFractTrs * size(spikeAveEp0,1)); % 30  % remove neurons with activity in <thSpTr trials.
nTrsWithSpike = sum(spikeAveEp0 > 0); % in how many trials each neuron had activity (remember this is average spike during ep).
NsFewTrActiv = nTrsWithSpike < thTrsWithSpike;


spikeAveEpAveTrs = nanmean(spikeAveEp0); % 1 x units % response of each neuron averaged across epoch ep and trials.
% thAct = 1e-3; % could be a good th for excluding neurons w too little activity.
nonActiveNs = spikeAveEpAveTrs < thAct;
fprintf('Number of non-active Ns= %d \n', sum(nonActiveNs))

% NsExcluded = NsFewTrActiv; % remove columns corresponding to neurons with activity in <thSpTr trials.
% NsExcluded = nonActiveNs; % remove columns corresponding to neurons with <thAct activity.
NsExcluded = logical(NsFewTrActiv + nonActiveNs);

spikeAveEp0(:, NsExcluded) = [];
fprintf('# included neuros = %d, fraction = %.3f\n', size(spikeAveEp0,2), size(spikeAveEp0,2)/size(traces_al_sm,2))
% figure; plot(max(spikeAveEp))
spikeAveEp0_sd = nanstd(spikeAveEp0);

%%
wNsHrLr = NaN(size(spikeAveEp0,2), numRand);
biasHrLr = NaN(1, numRand);
fractMisMatch = NaN(1, numRand);
avePerf = NaN(size(traces_al_sm,1), numRand);

%% Use equal number of trials for both HR and LR conditions.

extraTrs = setRandExtraTrs(find(choiceVec0==0), find(choiceVec0==1)); % find extra trials of the condition with more trials, so u can exclude them later.

choiceVec = choiceVec0;
% make sure choiceVec has equal number of trials for both lr and hr.
choiceVec(extraTrs) = NaN; % set to nan some trials (randomly chosen) of the condition with more trials so both conditions have the same number of trials.
trsExcluded = isnan(choiceVec);

fprintf('N trials for LR and HR = %d  %d\n', [sum(choiceVec==0), sum(choiceVec==1)])



% Make sure spikeAveEp has equal number of trials for both lr and hr.
spikeAveEp = spikeAveEp0;
spikeAveEp(extraTrs,:) = NaN; % set to nan some trials (randomly chosen) of the condition with more trials so both conditions have the same number of trials.
%%
xx = spikeAveEp0; % spikeAveEp0(extraTrs,:); % trials x units
yy = choiceVec0; % choiceVec0(extraTrs); % trials x 1

mskNan = isnan(choiceVec);
if windowAvgFlg
X = spikeAveEp(~mskNan, :); % spikeAveEp0(extraTrs,:); % trials x units
Y = choiceVec(~mskNan); % choiceVec0(extraTrs); % trials x 1
else
X = reshape(permute(traces_al_sm(ep, ~NsExcluded, ~mskNan), [1 3 2]),...
length(ep)*sum(~mskNan), sum(~NsExcluded));
Y = repmat(reshape(choiceVec(~mskNan), 1, sum(~mskNan)), length(ep), 1);
Y = Y(:);  
end

%% run SVM
cnam = [0,1]; % LR: negative ; HR: positive
% SVMModel = svmClassifierMS(X, Y, cnam);
if pcaFlg 
    [PCs, ~, l] = pca(X);
    numPCs = find(cumsum(l/sum(l))>0.99, 1, 'first');
    X_s = bsxfun(@plus, bsxfun(@minus, X, mean(X))*(PCs(:, 1:numPCs)*PCs(:, 1:numPCs)'), mean(X));
    SVMModel = fitcsvm(X_s, Y, 'standardize', 1, 'ClassNames', cnam, 'KernelFunction', 'linear'); % 'KernelFunction'. 'BoxConstraint'
else
    SVMModel = fitcsvm(X, Y, 'standardize', 1, 'ClassNames', cnam, 'KernelFunction', 'linear'); % 'KernelFunction'. 'BoxConstraint'
end
wNsHrLr(:,1) = SVMModel.Beta;
biasHrLr(1) = SVMModel.Bias;
            
fprintf('# neurons = %d\n', size(SVMModel.Mu, 2))
fprintf('# total trials = %d\n', SVMModel.NumObservations)
fprintf('# trials that are support vectors = %d\n', size(SVMModel.Alpha,1))

CVSVMModel = crossval(SVMModel);


%% compute label for all trs (not just the equal number trs that were
% randomly selected) and see how well it matches with the actual class.    
[label] = predict(SVMModel, xx); % predict(SVMModel, SVMModel.X);    
label(isnan(sum(xx,2))) = NaN;
fractMisMatch(1) = sum(abs(yy - label)>0) / sum(~isnan(yy - label));

%% see how well the SVM trained on our particular epoch can decode other time points.

corrClass = NaN(size(traces_al_sm,1), size(traces_al_sm,3)); % frames x trials

for itr = 1 : size(traces_al_sm,3)
    % u may wanna smooth traces_al_sm(:, ~NsExcluded, itr)
    a = traces_al_sm(:, ~NsExcluded, itr); % frames x neurons
    if any(isnan(a(:)))
        if ~all(isnan(a(:))), error('how did it happen?'), end
    elseif ~isnan(choiceVec0(itr))
        l = predict(SVMModel, a);
        corrClass(:, itr) = (l==choiceVec0(itr));
    end
end
% average performance (correct classification) across trials.
avePerf(:,1) = nanmean(corrClass, 2);  % frames x randomIters
    %% Make some plots to evaluate the SVM model
    if ~SVMModel.ConvergenceInfo.Converged, error('not converged!'), end
    
        fprintf('converged = %d\n', SVMModel.ConvergenceInfo.Converged)
        % SVMModel.NumObservations == size(choiceVec,1) - (length(extraTrs) + sum(isnan(choiceVec0))) % final number of trials
        % size(SVMModel.X,2) == size(spikeAveEp0,2) - sum(NsFewTrActiv) % final number of neurons
        
        if any(SVMModel.Prior ~= .5), error('The 2 conditions have non-equal number of trials!'), end
        %     fprintf('Prior probs = %.3f  %.3f\n', SVMModel.Prior) % should be .5 for both classes unless you used different number of trials for each class.
        
        % beta and bias (intercept) terms
        figure;
        
        subplot(221)
        plot(SVMModel.Beta)
        xlabel('Neuron')
        ylabel('SVM weight')
        title(sprintf('Bias = %.3f', SVMModel.Bias))
        
        % score and posterior probabilities of each trial belonging to the positive
        % class (HR in our case).
        [label, score] = resubPredict(SVMModel);
        [ScoreSVMModel, ScoreParameters] = fitPosterior(SVMModel); % or fitSVMPosterior
        [~, postProbs] = resubPredict(ScoreSVMModel);
        
        % score
        subplot(222)
        hold on, plot([0 size(score,1)], [0 0], 'handlevisibility', 'off')
        plot(score(:,2))
        xlabel('Trial')
        ylabel('Score') % of belonging to the positive class')
        xlim([1 SVMModel.NumObservations])
        
        % post prob
        subplot(224)
        hold on, plot([0 size(postProbs,1)], [.5 .5], 'handlevisibility', 'off')
        plot(postProbs(:,2))
        xlabel('Trial')
        ylabel('Posterior prob') % of belonging to the positive class')
        xlim([1 SVMModel.NumObservations])
        
        % label
        subplot(223), hold on
        title(['Classification error: ' num2str(sum(abs(SVMModel.Y-label))/length(label)*100) ' %'])
        plot(SVMModel.Y)
        plot(label)
        ylim([-1 2])
        xlabel('Trial')
        ylabel('Class')
        % legend('Actual', 'Model')
        xlim([1 SVMModel.NumObservations])
%%  quality relative to shuffles
       
classLossTrain = [];
classLossTest = [];
classLossChanceTrain = [];
classLossChanceTest = [];

wNsHrLr_s = [];
biasHrLr_s = [];
wNsHrLrChance = [];
biasHrLrChance = [];

for s = 1:100
    shflTrials = randperm(length(Y));
    X_s = X(shflTrials, :);
    Y_s = Y(shflTrials);
%%%%%%%% reduce features by PCA
    if pcaFlg
        [PCs, ~, l] = pca(X_s);
        numPCs = find(cumsum(l/sum(l))>0.99, 1, 'first');
        X_s = bsxfun(@plus, bsxfun(@minus, X_s, mean(X_s))*(PCs(:, 1:numPCs)*PCs(:, 1:numPCs)'), mean(X_s));
    end
     
%%%%%%%% data augmentation resampling
% % % %             mskNans = ~isnan(choiceVec);
% % % %             X_s = spikeAveEp(mskNans, :);
% % % %             Y_s = choiceVec(mskNans);
% % % %             numNeus = size(X_s, 2);
% % % %             X_sss = [];
% % % %             Y_sss = [];
% % % %             for ss = 1:3
% % % %                msk1 = Y_s == 1;
% % % %                X_s1 = X_s(msk1, :);
% % % %                Y_s1 = Y_s(msk1, :);
% % % %                X_s0 = X_s(~msk1, :);
% % % %                Y_s0 = Y_s(~msk1, :);
% % % %                X_ss = [];
% % % %                for n = 1:numNeus
% % % %                    X_ss(1:length(Y_s1), n) = X_s1(randi(length(Y_s1), length(Y_s1), 1), n);
% % % %                    X_ss(length(Y_s1)+(1:length(Y_s0)), n) = X_s0(randi(length(Y_s0), length(Y_s0), 1), n);
% % % %            
% % % %                end
% % % %                X_sss = [X_sss; X_ss];
% % % %                Y_sss = [Y_sss; Y_s1; Y_s0];
% % % %             end
% % % %             X_s = X_sss;
% % % %             Y_s = Y_sss;
%%%%%%%%%   


    SVMModel_s = fitcsvm(X_s, Y_s, 'standardize', 1, 'ClassNames', cnam); % Linear Kernel            
    classLossTrain(s) = mean(abs(Y_s-predict(SVMModel_s, X_s)));
    wNsHrLr_s(:, s) = SVMModel_s.Beta;
    biasHrLr_s(:, s) = SVMModel_s.Bias;
    
    CVSVMModel_s = crossval(SVMModel_s, 'kfold', 10); % CVSVMModel.Trained{1}: model 1 --> there will be KFold of these models. (by default KFold=10);
    classLossTest(s) = kfoldLoss(CVSVMModel_s); % Classification loss (by default the fraction of misclassified data) for observations not used for training
    Y_s_shfld = Y_s(randperm(length(Y_s)));
    SVMModelChance = fitcsvm(X_s, Y_s_shfld, 'standardize', 1, 'ClassNames', cnam); %  % Linear Kernel
    CVSVMModelChance = crossval(SVMModelChance, 'kfold', 10); % CVSVMModel.Trained{1}: model 1 --> there will be KFold of these models. (by default KFold=10);
    wNsHrLrChance(:, s) = SVMModelChance.Beta;
    biasHrLrChance(:, s) = SVMModelChance.Bias;
    
    classLossChanceTrain(s) = mean(abs(Y_s_shfld-predict(SVMModelChance, X_s)));
    classLossChanceTest(s) = kfoldLoss(CVSVMModelChance); % Classification loss (by default the fraction of misclassified data) for observations not used for training
    s
end
classLoss = mean(classLossTest);
fprintf('Average cross-validated classification error = %.3f\n', (classLoss))
figure;
subplot(211)
hold on
%         hd = hist(classLossTrain, 0:0.02:1);
hc = hist(classLossChanceTrain, 0:0.02:1);
bar(0:0.02:1, hc, 'facecolor', 0.5*[1 1 1], 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
%         bar(0:0.02:1, hd, 'facecolor', 'r', 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
plot(mean(classLossTrain), 0, 'ko','markerfacecolor', 'r', 'markersize', 6)
plot(mean(classLossChanceTrain), 0, 'ko','markerfacecolor', 0.5*[1 1 1], 'markersize', 6)
ylabel('Count')
xlabel('training loss')
xlim([0 1])
legend('shuffled', 'data', 'location', 'northwest')
legend boxoff

subplot(212)
[~, p]= ttest2(classLossTest, classLossChanceTest, 'tail', 'left', 'vartype', 'unequal');
hold on
hd = hist(classLossTest, 0:0.02:1);
hc = hist(classLossChanceTest, 0:0.02:1);
bar(0:0.02:1, hc, 'facecolor', 0.5*[1 1 1], 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
bar(0:0.02:1, hd, 'facecolor', 'r', 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
plot(mean(classLossTest), 0, 'ko','markerfacecolor', 'r', 'markersize', 6)
plot(mean(classLossChanceTest), 0, 'ko','markerfacecolor', 0.5*[1 1 1], 'markersize', 6)
ylabel('Count')
title(['p-value lower tail = ' num2str(p)])
xlabel('cross-validation loss')
xlim([0 1])
legend('shuffled', 'data', 'location', 'northwest')
legend boxoff
  
fprintf('Average cross-validated classification error = %.3f\n', mean(classLoss))
%         cl(i) = classLoss;
%         end
% Estimate cross-validation predicted labels and scores.
% For every fold, kfoldPredict predicts class labels for in-fold
% observations using a model trained on out-of-fold observations.
[elabel, escore] = kfoldPredict(CVSVMModel);

% Estimate the out-of-sample posterior probabilities
[ScoreCVSVMModel, ScoreParameters] = fitSVMPosterior(CVSVMModel);
[~, epostp] = kfoldPredict(ScoreCVSVMModel);
% How claassLoss is computed? I think: classLoss = 1 - mean(label == elabel)
diff([classLoss, mean(label ~= elabel)])


%% Average b across all iters (bagging : bootstrap aggregation)

% figure; imagesc(wNsHrLr)
% figure; errorbar(1:size(wNsHrLr,1), mean(wNsHrLr, 2), std(wNsHrLr, [], 2), 'k.')

bLen = sqrt(sum(wNsHrLr.^2)); % norm of wNsHrLr for each rand
% figure; plot(bLen)
wNsHrLrNorm = bsxfun(@rdivide, wNsHrLr, bLen); % normalize b of each rand by its vector length 
wNsHrLrAve = mean(wNsHrLrNorm, 2); % average of normalized b across all rands.
wNsHrLrAve = wNsHrLrAve / norm(wNsHrLrAve); % normalize it so the final average vector has norm of 1.
% figure; plot(bNsHrLrAve)
% figure; errorbar(1:size(wNsHrLr,1), mean(wNsHrLrNorm, 2), std(wNsHrLrNorm, [], 2), 'k.')



%% compute fraction of mismatch in classification

xx = spikeAveEp0;
yy = choiceVec0; % (extraTrs); % choiceVec0

% scale xx
x = bsxfun(@minus, xx, nanmean(xx)); % (x-mu)
x = bsxfun(@rdivide, x, nanstd(xx)); % (x-mu)/sigma
x = x / SVMModel.KernelParameters.Scale; % scale is 1 unless u've changed it in svm model.


% compute score on the aggregate of all iters of svm.
s = x * wNsHrLrAve + nanmean(biasHrLr); % score = x*beta + bias % bias should be added too but I'm not sure if averating bias across iters is the right thing to do. Also it seems like not including the bias term gives better separaton of hr and lr.

label = s;
label(s>0) = 1; % score<0 will be lr and score>0 will be hr, so basically threshold is 0.... 
label(s<0) = 0;
if sum(s==0)>0, error('not sure what to assign as label when score is 0!'), end

fractMisMatchFinal = sum(abs(yy - label)>0) / sum(~isnan(yy - label));

[nanmean(fractMisMatch) fractMisMatchFinal]


% compare with fractMisMatch on each iter ... see if doing several
% iters helped w better prediction:
figure; hold on,
plot([0 length(fractMisMatch)], [fractMisMatchFinal fractMisMatchFinal])
plot(fractMisMatch)
title(sprintf('%.3f  %.3f', nanmean(fractMisMatch), fractMisMatchFinal))


%{
%% plot bias
figure('name', 'bias term'); subplot(211), plot(biasHrLr)
subplot(212),  errorbar( mean(biasHrLr), std(biasHrLr), 'k.')
%}


%%
%% Plots
%%


% make some plots related to the decoder

%% Simple averaging of neural responses for HR and LR trials. 
% decide you training window based on this plot!

% traces = traces_al_sm(:, ~NsExcluded, :); % equal number of trs for both conds
% av1 = nanmean(nanmean(traces(:,:, choiceVec==1), 3), 2);
% av2 = nanmean(nanmean(traces(:,:, choiceVec==0), 3), 2);

traces = traces_al_sm(:, ~NsExcluded, :); % analyze all trials
av1 = nanmean(nanmean(traces(:,:, choiceVec0==1), 3), 2);
av2 = nanmean(nanmean(traces(:,:, choiceVec0==0), 3), 2);
% average for HR vs LR stimulus.
% av1 = nanmean(nanmean(traces(:,:, stimrate > cb), 3), 2);
% av2 = nanmean(nanmean(traces(:,:, stimrate < cb), 3), 2);

pb = round((- eventI_stimOn)*frameLength);
pe = round((size(traces,1) - eventI_stimOn)*frameLength);
st = round((epStart - eventI_stimOn)*frameLength);
en = round((epEnd - eventI_stimOn)*frameLength);
mn = min([av1;av2]);
mx = max([av1;av2]);
figure; 
subplot(223), hold on
plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI_stimOn eventI_stimOn]
plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]
plot(time_aligned_stimOn, av1), plot(time_aligned_stimOn, av2)
xlabel('Time since stim onset (ms)')
ylabel('Average neural responses')
xlim([pb pe])


%% Weighted average of neurons for each trial, using svm weights trained on a particular epoch (ep).

% see how well that particular epoch can decode other epochs.

% include all trs (not just the random equal trs) and use the average b  across all iters.
frameTrProjOnBeta = einsum(traces_al_sm(:, ~NsExcluded, :), wNsHrLrAve, 2, 1); % (fr x u x tr) * (u x 1) --> (fr x tr)
% below is exactly same as the above. doing projection on the average
% weight is same as averaging projection. (each individually projected onto
% its svm vector)
% frameTrProjOnBeta = nanmean(einsum(traces_al_sm(:, ~NsExcluded, :), wNsHrLrNorm, 2, 1), 3); % get the projection on the svm vector of each iter and then compute the ave across iters
size(frameTrProjOnBeta) % frames x trials
frameTrProjOnBeta_hr = frameTrProjOnBeta(:, choiceVec0==1); % frames x trials % HR
frameTrProjOnBeta_lr = frameTrProjOnBeta(:, choiceVec0==0); % frames x trials % LR


av1 = nanmean(frameTrProjOnBeta_hr, 2);
av2 = nanmean(frameTrProjOnBeta_lr, 2);
mn = min([av1;av2]);
mx = max([av1;av2]);
% figure; 
subplot(221), hold on
plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI_stimOn eventI_stimOn]
plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]
plot(time_aligned_stimOn, av1), plot(time_aligned_stimOn, av2)
xlabel('Time since stim onset (ms)')
ylabel({'Weighted average of', 'neural responses'})
xlim([pb pe])

% frameTrProjOnBeta = einsum(traces_al_sm(:, ~NsExcluded, ~trsExcluded), SVMModel.Beta, 2, 1); % (fr x u x tr) * (u x 1) --> (fr x tr)
% size(frameTrProjOnBeta) % frames x trials
% frameTrProjOnBeta_hr = frameTrProjOnBeta(:, SVMModel.Y==1); % frames x trials % HR
% frameTrProjOnBeta_lr = frameTrProjOnBeta(:, SVMModel.Y==0); % frames x trials % LR


%% Decoder performance at each time point (decoder trained on epoch ep)

top = nanmean(avePerf, 2); % average performance across all iters.
mn = min(top(:));
mx = max(top(:));
% figure; 
subplot(222), hold on
plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI_stimOn eventI_stimOn]
plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]
plot(time_aligned_stimOn, top) % average across all iters
xlabel('Time since stim onset (ms)')
ylabel('Correct classification')
xlim([pb pe])


%% Weighted average of neural responses for all trials and its distribution for hr vs lr trials. U compute this on normalized x bc score=0 as the threshold for hr and lr is defined on the projection of normalized x onto the weights.
% comapre aggregate beta with beta from only 1 run of svm training.
xx = spikeAveEp0;
yy = choiceVec0; % (extraTrs); % choiceVec0

% scale xx
x = bsxfun(@minus, xx, nanmean(xx)); % (x-mu)
x = bsxfun(@rdivide, x, nanstd(xx)); % (x-mu)/sigma
x = x / SVMModel.KernelParameters.Scale; % scale is 1 unless u've changed it in svm model.


%%% Distributions
% use Beta computed from one iter
weightedAveNs_allTrs = x * SVMModel.Beta; % trials x 1

% use aggregate Beta (averaged across all iters)
weightedAveNs_allTrs = x * wNsHrLrAve; % + nanmean(biasHrLr); % score = x*beta + bias % bias should be added too but I'm not sure if averating bias across iters is the right thing to do. Also it seems like not including the bias term gives better separaton of hr and lr.


weightedAveNs_hr = weightedAveNs_allTrs(yy==1);
weightedAveNs_lr = weightedAveNs_allTrs(yy==0);
%{
weightedAveNs_allTrs = SVMModel.X * SVMModel.Beta; % trials x 1 % this is wrong u have to scale (center and normalize) SVMModel.X, bc SVMModel.Beta is for scaled data.
weightedAveNs_hr = weightedAveNs_allTrs(SVMModel.Y==1);
weightedAveNs_lr = weightedAveNs_allTrs(SVMModel.Y==0);
%}

% Compare the dist of weights for HR vs LR
[nh, eh] = histcounts(weightedAveNs_hr, 'normalization', 'probability');
[nl, el] = histcounts(weightedAveNs_lr, 'normalization', 'probability');

% figure; 
subplot(224), hold on
plot(eh(1:end-1), nh)
plot(el(1:end-1), nl)
legend('HR','LR')
xlabel('Weighted average of neurons for epoch ep')
ylabel('Fraction of trials')

% cross point of 2 gaussians when they have similar std:
mu0 = mean(weightedAveNs_hr);
mu1 = mean(weightedAveNs_lr);
sd0 = std(weightedAveNs_hr);
sd1 = std(weightedAveNs_lr);
threshes = (sd0 * mu1 + sd1 * mu0) / (sd1 + sd0);
plot([threshes threshes], [0 .5], ':')


%% Compare svm weights with random weights

popClassifierSVM_rand


%% Compare SVM weights with ROC choicePref

% popClassifierSVM_choicePref









