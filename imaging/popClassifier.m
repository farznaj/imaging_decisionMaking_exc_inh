% Remember: you can use the script svmUnderstandIt to understand how some
% of the matlab functions related to SVM classification work.

stMs = round(500/frameLength);
enMs = round(900/frameLength);

thAct = 1e-3; % could be a good th for excluding neurons w too little activity.
numRand = 1; % 50; 100; % you tried values between [50 500], at nrand=500 mismatches (computed on the b averaged across all iters) are actually worse compared to the average mismatch computed from single runs.  nrand=200 seems to be a good choice.
doplots = false;


method = 'svm'; % 'logisticRegress';  % classification method for neural population analysis.
rng(0, 'twister'); % Set random number generation for reproducibility


%% Set Y: the response vector

choiceVec0 = allResp_HR_LR';  % trials x 1;  1 for HR choice, 0 for LR choice.
choiceVec0(outcomes~=1) = NaN; % analyze only correct trials.
% choiceVec(trs2rmv) = NaN; % choiceVec(isnan(stimrate)) = NaN;
fprintf('N trials for LR and HR = %d  %d\n', [sum(choiceVec0==0), sum(choiceVec0==1)])

%{
d = diff(trialNumbers);
u = unique(d);
if length(u)>1
    error('A gap in imaged trials. Trial history analysis will be problematic! Look into it!')
end

X = X(trialNumbers,:);
%}


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
% spikeAveEp = spikeAveEp * 1000 / frameLength; % spikes/sec : average number of spikes in 1000ms.


%% Remove (from X) neurons with average activity during epoch ep in too few trials. or with too little average activity during epoch ep and averaged across all trials.
%
thMinFractTrs = .05; %.01; % a neuron must be active in >= .1 fraction of trials to be used in the population analysis.
thTrsWithSpike = 3; ceil(thMinFractTrs * size(spikeAveEp0,1)); % 30  % remove neurons with activity in <thSpTr trials.
nTrsWithSpike = sum(spikeAveEp0 > 0); % in how many trials each neuron had activity (remember this is average spike during ep).
% figure; plot(nTrsWithSpike)
% hold on, plot([1 length(nTrsWithSpike)],[thTrsWithSpike  thTrsWithSpike])
NsFewTrActiv = nTrsWithSpike < thTrsWithSpike;
%

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
xx = spikeAveEp0; % spikeAveEp0(extraTrs,:); % trials x units
yy = choiceVec0; % choiceVec0(extraTrs); % trials x 1
    

%%
wNsHrLr = NaN(size(spikeAveEp0,2), numRand);
biasHrLr = NaN(1, numRand);
fractMisMatch = NaN(1, numRand);
avePerf = NaN(size(traces_al_sm,1), numRand);

for r = 1:numRand
    
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
    
    
    %     %% Remove (from X) neurons with average activity during epoch ep in too few trials. or with too little average activity during epoch ep and averaged across all trials.
    
    
    %% Classify neural population responses for HR and LR choices

    % wNsHrLr: neurons x numRands
    switch method
        case 'logisticRegress' % Logistic regression on population neural responses
            [B, deviance, stats] = glmfit(spikeAveEp, choiceVec, 'binomial', 'estdisp', 'on'); % remember the 1st terms is constant.
            % B = mnrfit(spikeAveEp, choiceVec+1)
            % B(1), figure; plot(B(2:end))
            wNsHrLr(:,r) = B(2:end);
            biasHrLr(r) = B(1);
            
        case 'svm'
            cnam = [0,1]; % LR: negative ; HR: positive
            SVMModel = fitcsvm(spikeAveEp, choiceVec, 'standardize', 1, 'ClassNames', cnam); % 'KernelFunction'. 'BoxConstraint'
            wNsHrLr(:,r) = SVMModel.Beta;
            biasHrLr(r) = SVMModel.Bias;
            % b = SVMModel.Bias; figure; plot(SVMModel.Beta);
            
            fprintf('# neurons = %d\n', size(SVMModel.Mu, 2))
            fprintf('# total trials = %d\n', SVMModel.NumObservations)
            fprintf('# trials that are support vectors = %d\n', size(SVMModel.Alpha,1))
    end
    
    
    %% are neurons with higher variability more heavily used by the classifier?
%     corrcoef(SVMModel.Beta, SVMModel.Sigma)
    
    
    %% compute label for all trs (not just the equal number trs that were
    % randomly selected) and see how well it matches with the actual class.    
    [label] = predict(SVMModel, xx); % predict(SVMModel, SVMModel.X);    
    label(isnan(sum(xx,2))) = NaN;
    fractMisMatch(r) = sum(abs(yy - label)>0) / sum(~isnan(yy - label));


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
    avePerf(:,r) = nanmean(corrClass, 2);  % frames x randomIters


    %% Make some plots to evaluate the SVM model
    if ~SVMModel.ConvergenceInfo.Converged, error('not converged!'), end
    
    if doplots        
        %%
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
        plot(SVMModel.Y)
        plot(label)
        ylim([-1 2])
        xlabel('Trial')
        ylabel('Class')
        % legend('Actual', 'Model')
        xlim([1 SVMModel.NumObservations])
        
%         clear cl
%         for i = 1:100
        % Cross validation: evaluate the Classification Error of the SVM classifier
        CVSVMModel = crossval(SVMModel, 'kfold', 10); % CVSVMModel.Trained{1}: model 1 --> there will be KFold of these models. (by default KFold=10);
        
        classLoss = kfoldLoss(CVSVMModel); % Classification loss (by default the fraction of misclassified data) for observations not used for training
        fprintf('Average cross-validated classification error = %.3f\n', classLoss)
%         cl(i) = classLoss;
%         end
        % Estimate cross-validation predicted labels and scores.
        % For every fold, kfoldPredict predicts class labels for in-fold
        % observations using a model trained on out-of-fold observations.
        [elabel, escore] = kfoldPredict(CVSVMModel);
        
        % Estimate the out-of-sample posterior probabilities
        [ScoreCVSVMModel, ScoreParameters] = fitSVMPosterior(CVSVMModel);
        [~, epostp] = kfoldPredict(ScoreCVSVMModel);
        
        subplot(222)
        plot(escore(:,2)), legend('Trained', 'Tested')
        subplot(224)
        plot(epostp(:,2))
        subplot(223)
        plot(elabel)
        title(sprintf('Class Loss = %.3f\n', classLoss))
        
        % How claassLoss is computed? I think: classLoss = 1 - mean(label == elabel)
        diff([classLoss, mean(label ~= elabel)])
    end
    
end


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
s = x * wNsHrLrAve; % + nanmean(biasHrLr); % score = x*beta + bias % bias should be added too but I'm not sure if averating bias across iters is the right thing to do. Also it seems like not including the bias term gives better separaton of hr and lr.

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

