function popClassifier0(alldata, alldataSpikesGood, outcomes, allResp_HR_LR, frameLength, ...
    timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, ...
    neuronType, trialHistAnalysis, numShuffs, nPreFrames, nPostFrames, alignedEvent, onlyCorrect, iTiFlg, prevSuccessFlg, vec_iti)
% This is the 1st popClassifier code that I wrote. The difference with
% popClassifier is that here you remove extraTrs (so hr and lr have the
% same number of trials) at the beginning of the code and outside the
% numShuffs loop. You also do PCA outside the numShuffs loop. But in
% popClassifier (the newer code) you set extraTrs inside numShuffs loop.
% The advantage is that if extraTrs includes lots of trials, you will get
% more random sets of trials in each numShuffs (instead of sticking to one
% signle set of trials and shuffling them). However because you compute PCA
% inside numShuffs the newer code is slower.
%
% This is the main function for doing SVM analysis on calcium imaging data.
% Run imaging_prep_analysis to get the required input vars.
%
% Predicting choice given the population responses (trials x neurons) at an
% epoch during the stimulus.
%
% You can use the script svmUnderstandIt to understand how some
% of the matlab functions related to SVM classification work.
%
% Author: Farzaneh Najafi with modifications from Gamal Elsayed.
% Jan 2016
%
home
fprintf('========= SVM analysis started =========\n')


%% Set initial variables
%{
neuronType = 2; % 0: excitatory, 1: inhibitory, 2: all types.
trialHistAnalysis = 0; % more parameters are specified in popClassifier_trialHistory.m
    iTiFlg = 1; % 0: short ITI, 1: long ITI, 2: all ITIs.
    prevSuccessFlg = true; % true previous sucess trials; false: previous failure.
    vec_iti = [0 9 30]; % [0 10 30]; %[0 6 9 12 30]; % [0 7 30]; % [0 10 30]; % [0 6 9 12 30]; % use [0 40]; if you want to have a single iti bin and in conventioinal analysis look at the effect of current rate on outcome.
    
numShuffs = 10; % 100 % number of iterations for getting CV models and shuffled data.
onlyCorrect = 0; % If 1, analyze only correct trials.

% Set nPre and nPost to nan to make sure frames before and after
% alignedEvent don't have any other events.
% Set to [] to include all exisiting frames before and after the
% alignedEvent (regardless of whether they include other frames or not).
% Set to any other value if you wan to manually specify the number of frames
% before and after alignedEvent in the aligned traces (ie traces_al_sm).
nPreFrames = []; % nan; %
nPostFrames = []; % nan; % 

% alignedEvent: 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward', 'commitIncorrResp'
if trialHistAnalysis
    alignedEvent = 'initTone';
else
    alignedEvent = 'commitIncorrResp'; 'reward'; 'goTone';  % 'stimOn'; % what event align traces on. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward'
end
%}

%%%%%%%%%% Define training epoch (ie the epoch over which neural activity will be
% averaged and on which SVM will be trained): 
% epStart: 1st frame (index of frames in traces_al_sm) of the training epoch.
% epEnd: last frame (index of frames in traces_al_sm) of the training epoch.
% Alternatively, you can define frame indeces relative to alignedEvent by
% setting epStartRel2Event and epEndRel2Event.
% epStartRel2Event: 1st frame (after alignedEvent) of the training epoch. 
% epEndRel2Event: last frame (after alignedEvent) of the training epoch.

% This helps to figure out what to use as epEnd
defaultPrePostFrames = 2; % default value for nPre and postFrames, if their computed values < 1.
[~, ~, ~, nPre, nPost] = alignTraces_prePost_allCases(alignedEvent, [], {alldata.frameTimes}, ...
    frameLength, defaultPrePostFrames, [], [], timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, [], 1, nan, nan, 1);
fprintf('%i = number of post event frames without any other events.\n', nPost)
warning('Double check the start and end point of the training epoch!')
% fprintf('%i = event frame if nPreFrames was set to NaN.\n%i = number of post event frames without any other events.\n', nPre+1, nPost)

clear epStart epEnd
if trialHistAnalysis
    disp('Training epoch = all frames before alingedEvent (including alignedEvent).')
    % index of frames in traces_al_sm
    epStart = 1; %3; %4;
    epEnd = nan; %epStart + round(200/frameLength); % if nan, epEnd = time of alignedEvent.
    
elseif strcmp(alignedEvent, 'stimOn')
    disp('Training epoch = 600-800ms after stimulus onset.')
    % index of frames after alignedEvent (if epStartRel2Event=1, epStart
    % will be 1 frame after alignedEvent).
    epStartRel2Event = round(600/frameLength); % the start point of the epoch relative to alignedEvent for training SVM. (500ms)
    epEndRel2Event = floor(800/frameLength); % the end point of the epoch relative to alignedEvent for training SVM. (700ms)
else
    disp('Training epoch = all frames after alingedEvent and before the next event.')
    epStartRel2Event = 1;
    if ~isempty(nPost)
        epEndRel2Event = nPost;
    else
        disp('... there is no next event, so training epoch = all frames after alignedEvent')
        epEndRel2Event = nan;
    end
end


pcaFlg = true;
windowAvgFlg = true;
shuffleTrsForIters = 1; % if 1, in the iterations you will shuffle trials where you set CV SVM models and shuffled data, you will shuffle trials (this is different from the shuffling for computing chance distributions).
usePooledWeights = 0; % if 1, weights will be pooled across different shuffles and then a single weight vector will be used to compute projections. If 0, weights of each model will be used for making projections of that model.


plot_rand_choicePref = 0; % if 1, plots related to random decoders (similar to NN paper) will be made. Also plots that compare weights for each neuron with its measure of choicePref to see if higher weights are associated with higher ROC measure of choicePref.
doplot_svmBasics = 0; % if 1, it will plot a figure showing weights, scores, posterior probability and labels for each neuron.

% thAct = quantile(spikeAveEpAveTrs, .1); % this is what you are using now,
% ie the lowest 10 percentile of average activity during ep across all
% neurons defines the threshold for identifying non-active neurons (to be
% excluded).
% thAct = 1e-3; % Perhaps for MCMC: could be a good th for excluding neurons w too little activity.


%%%
% numRand = 1; % 50; 100; % you tried values between [50 500], at nrand=500 mismatches (computed on the b averaged across all iters) are actually worse compared to the average mismatch computed from single runs.  nrand=200 seems to be a good choice.
doplots = true;
rng(0, 'twister'); % Set random number generation for reproducibility
% method = 'svm'; % 'logisticRegress';  % classification method for neural population analysis.


%% Align traces on particular trial events

% remember traces_al_sm has nan for trs2rmv as well as trs in alignedEvent that are nan.

traces = alldataSpikesGood; % alldataSpikesGoodExc; % alldataSpikesGoodInh; % alldataSpikesGood;  % traces to be aligned.
% alignedEvent = 'stimOn'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward'
dofilter = false; % true;
traceTimeVec = {alldata.frameTimes};

% % set nPre and nPost to nan if you want to go with the numbers that are based on eventBef and eventAft.
% % set to [] to include all frames before and after the alignedEvent.
% nPreFrames = nan;[];4;
% nPostFrames = nan;[];10;
% alignedEvent = 'reward';    % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward', 'commitIncorrResp'
[traces_al_sm, time_aligned, eventI] = alignTraces_prePost_filt(traces, traceTimeVec, alignedEvent, frameLength, dofilter, ...
    timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, nPreFrames, nPostFrames);

switch neuronType
    case 0 % only excitatory
%         traces_al_sm(:, good_inhibit, :) = NaN;
    traces_al_sm = traces_al_sm(:, good_excit, :);
    case 1 % only inhibitory
%         traces_al_sm(:, good_excit, :) = NaN;
    traces_al_sm = traces_al_sm(:, good_inhibit, :);
end


% set to nan those trials in outcomes and allRes that are nan in traces_al_sm
a = find(sum(sum(~isnan(traces_al_sm),1),3), 1);
allTrs2rmv = find(squeeze(sum(isnan(traces_al_sm(:,a,:)))));
% fprintf('%d= final number of NaN trials in the aligned traces\n', length(allTrs2rmv))

outcomes(allTrs2rmv) = NaN;
% allResp(allTrs2rmv) = NaN;
allResp_HR_LR(allTrs2rmv) = NaN;


%% Start setting Y: the response vector

if trialHistAnalysis
    popClassifier_trialHistory
else
    choiceVec0 = allResp_HR_LR';  % trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.
    
    if onlyCorrect
        fprintf('Analyzing only correct trials.\n')
        choiceVec0(outcomes~=1) = NaN; % analyze only correct trials.
    else
        fprintf('Analyzing both correct and incorrect trials.\n')
    end
end

fprintf('#trials for LR and HR choices = %d  %d\n', [sum(choiceVec0==0), sum(choiceVec0==1)])


%% Set ep: the epoch that SVM will be trained on. Average activity of neurons during this epoch will be used for training.

% Set the epoch of neural responses that you want to analyze.
if ~exist('epStart', 'var') || ~exist('epEnd', 'var')
    
    nPostFrs = size(traces_al_sm,1) - (eventI-1); % number of frames after the stimOn (including stimOn frame) in the aligned traces.
    
    if eventI+epStartRel2Event < 1
        warning('Resetting epStart to 1!')
    end
    epStart = max(1, eventI+epStartRel2Event); %20; round(500/frameLength); % eventI_stimOn; % start of the epoch in frames (not ms)
    
    if size(traces_al_sm,1) < epStart
        warning('Resetting epStart to number of frames in traces_al_sm!')
    end
    epStart = min(size(traces_al_sm,1), epStart); % make sure
    
    epEnd = min(eventI+epEndRel2Event, eventI+nPostFrs-1); % eventI_stimOn+epLen-1; % end of the epoch in frames (not ms)
    
    % below is to make sure epEndRel2Event is not after stimOffset, ie you
    % don't train SVM on a window that includes post-stimOffset responses.
    % (Lets say you set epEndRel2Event such that training window's last
    % timepoint will be 1100ms after stim onset, even though stimulus was
    % only on for 1000ms. By running the codes below you will make sure
    % that training window won't go beyong 1000ms.)
    if strcmp(alignedEvent, 'stimOn')
        minStimFrs = floor(min(stimdur*1000)/frameLength); % minimum stimdur of all trials in frames    
        if minStimFrs < epEndRel2Event
            warning('Resetting epEndRel2Event to min of stimDur!')
        end
        epLen = min(nPostFrs, minStimFrs); % length of the epoch we are going to analyze.

        epEnd = min(eventI+epEndRel2Event, eventI+epLen-1); % eventI_stimOn+epLen-1; % end of the epoch in frames (not ms)
    end
end

if isnan(epEnd)
    epEnd = eventI;
end

ep = epStart : epEnd; % frames in traces_al_sm that will be used for analysis. % for now lets use spike counts over the entire stim frames.
fprintf('%d : %d = Training epoch (frame indeces in traces_al_sm)\n', ep(1), ep(end))
% fprintf([repmat('%d ', 1, length(ep)), '= Training epoch (frame indeces in traces_al_sm)\n'], ep)


%% Start setting X: the predictor matrix (trials x neurons) that shows average of spikes for a particular epoch for each trial and neuron.

% Compute average of spikes per frame during epoch ep.
spikeAveEp0 = squeeze(nanmean(traces_al_sm(ep,:,:)))'; % trials x units.

% smooth the traces (moving average) using a window of size ep.
filtered0 = boxFilter(traces_al_sm, length(ep), 1, 0);
% figure; plot(traces_al_sm(:,4,13))
% hold on, plot(filtered(:,4,13))
% spikeAveEp0(13,4)

spikeAveEp00 = spikeAveEp0; % save it before excluding any neurons.


%% Identify neurons that are very little active.
% Little activity = neurons that are active in few trials. Also neurons that
% have little average activity during epoch ep across all trials.

% Set nonActiveNs, ie neurons whose average activity during ep is less than thAct.
spikeAveEpAveTrs = nanmean(spikeAveEp0); % 1 x units % response of each neuron averaged across epoch ep and trials.
thAct = quantile(spikeAveEpAveTrs, .1);
warning('You are using .1 quantile of average activity during ep across all neurons as threshold for identifying non-active neurons. This is arbitrary and needs evaluation!')
% thAct = 1e-3; % could be a good th for excluding neurons w too little activity.
nonActiveNs = spikeAveEpAveTrs < thAct;
fprintf('%d= # neurons with ave activity in ep < %.4f\n', sum(nonActiveNs), thAct)


% Set NsFewTrActiv, ie neurons that are active in very few trials (by active I mean average activity during epoch ep)
% thMinFractTrs = .05; %.01; % a neuron must be active in >= .1 fraction of trials to be used in the population analysis.
thTrsWithSpike = 3; % ceil(thMinFractTrs * size(spikeAveEp0,1)); % 30  % remove neurons with activity in <thSpTr trials.

% nTrsWithSpike = sum(spikeAveEp0 > 0); % in how many trials each neuron
% had activity (remember this is average spike during ep). 
% Remember the zero threshold used above really only makes sense if sikes
% were infered using the MCMC method, otherwise in foopsi, S has arbitrary
% values and unless a traces is all NaNs (which should not happen)
% spikeAveEp0 will be > 0 all the time.

nTrsWithSpike = sum(spikeAveEp0 > thAct);
NsFewTrActiv = nTrsWithSpike < thTrsWithSpike;
fprintf('%d= # neurons that are active in less than %i trials.\n', sum(NsFewTrActiv), thTrsWithSpike)


% Now set the final NxExcluded: (neurons to exclude)
% NsExcluded = NsFewTrActiv; % remove columns corresponding to neurons with activity in <thSpTr trials.
% NsExcluded = nonActiveNs; % remove columns corresponding to neurons with <thAct activity.
NsExcluded = logical(NsFewTrActiv + nonActiveNs);

a = size(spikeAveEp0,2) - sum(NsExcluded);
fprintf('included neuros= %d; total neuros= %d; fract= %.3f\n', a, size(spikeAveEp0,2), a/size(spikeAveEp0,2))



%% Remove neurons that are very little active.
% Remove (from X) neurons that are active in few trials. Also neurons that
% have little average activity during epoch ep across all trials.

spikeAveEp0(:, NsExcluded) = [];
% fprintf('# included neuros = %d, fraction = %.3f\n', size(spikeAveEp0,2), size(spikeAveEp0,2)/size(traces_al_sm,2))
% figure; plot(max(spikeAveEp))
spikeAveEp0_sd = nanstd(spikeAveEp0);

filtered1 = filtered0(:, ~NsExcluded, :);


%% Use equal number of trials for both HR and LR conditions.

extraTrs = setRandExtraTrs(find(choiceVec0==0), find(choiceVec0==1)); % find extra trials of the condition with more trials, so u can exclude them later.

choiceVec = choiceVec0;
% make sure choiceVec has equal number of trials for both lr and hr.
choiceVec(extraTrs) = NaN; % set to nan some trials (randomly chosen) of the condition with more trials so both conditions have the same number of trials.
% trsExcluded = isnan(choiceVec);

fprintf('Final # trials for LR and HR = %d  %d\n', [sum(choiceVec==0), sum(choiceVec==1)])
fprintf('%i= #trials excluded to have same trNums for both HR & LR\n', length(extraTrs))


% Make sure spikeAveEp has equal number of trials for both lr and hr.
spikeAveEp = spikeAveEp0;
spikeAveEp(extraTrs,:) = NaN; % set to nan some trials (randomly chosen) of the condition with more trials so both conditions have the same number of trials.


%% Set X and Y : predictor matrix and response vector.

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
% Y(Y==0) = -1; % doesn't make a differece.

non_filtered = traces_al_sm(:, ~NsExcluded, ~mskNan);
filtered = filtered1(:,:,~mskNan);


if pcaFlg    
    % X
    [PCs, ~, l] = pca(X);
    numPCs = find(cumsum(l/sum(l))>0.99, 1, 'first');
    fprintf('Number of PCs:\n   %i : total\n   %i : >.99 variance\n', length(l), numPCs)

    X_s = bsxfun(@plus, bsxfun(@minus, X, mean(X))*(PCs(:, 1:numPCs)*PCs(:, 1:numPCs)'), mean(X));
    
    
    % filtered
    filtered_s = NaN(size(filtered));    
    for fr = 1:size(filtered,1)
        Xf = squeeze(filtered(fr,:,:))';        
        [PCs_f, ~, l] = pca(Xf);
        numPCs_f = find(cumsum(l/sum(l))>0.99, 1, 'first');        
        filtered_s(fr,:,:) = bsxfun(@plus, bsxfun(@minus, Xf, mean(Xf))*(PCs_f(:, 1:numPCs_f)*PCs_f(:, 1:numPCs_f)'), mean(Xf))';
    end    
end

% trsInChoiceVecUsedInIterS = repmat(find(~mskNan), 1, numShuffs);


%% Run SVM (fitcsvm)

cnam = [0,1]; % LR: negative ; HR: positive
wNsHrLr = NaN(size(spikeAveEp0,2), 1); %numRand);
biasHrLr = NaN(1, 1); %numRand);
fractMisMatch_allTrs = NaN(1, 1); %numRand);
avePerf = NaN(size(traces_al_sm,1), 1); %numRand);

% SVMModel = svmClassifierMS(X, Y, cnam);
if pcaFlg
    SVMModel = fitcsvm(X_s, Y, 'standardize', 1, 'ClassNames', cnam, 'KernelFunction', 'linear'); % 'KernelFunction'. 'BoxConstraint'
else
    SVMModel = fitcsvm(X, Y, 'standardize', 1, 'ClassNames', cnam, 'KernelFunction', 'linear'); % 'KernelFunction'. 'BoxConstraint'
end
wNsHrLr(:,1) = SVMModel.Beta;
biasHrLr(1) = SVMModel.Bias;

fprintf('# neurons = %d\n', size(SVMModel.Mu, 2))
fprintf('# total trials = %d\n', SVMModel.NumObservations)
fprintf('# trials that are support vectors = %d\n', size(SVMModel.Alpha,1))


%%% A few checks :

if ~SVMModel.ConvergenceInfo.Converged, error('not converged!'), end

%%% fprintf('converged = %d\n', SVMModel.ConvergenceInfo.Converged)
% SVMModel.NumObservations == size(choiceVec,1) - (length(extraTrs) + sum(isnan(choiceVec0))) % final number of trials
% size(SVMModel.X,2) == size(spikeAveEp0,2) - sum(NsFewTrActiv) % final number of neurons

if any(SVMModel.Prior ~= .5), error('The 2 conditions have non-equal number of trials!'), end
%     fprintf('Prior probs = %.3f  %.3f\n', SVMModel.Prior) % should be .5 for both classes unless you used different number of trials for each class.



%% Compute label (class) for all trials and see how well it matches the actual class.

% Remember only the following fraction: 
% length(extraTrs) / sum(~isnan(choiceVec0)) 
% of trials in xx don't exist in X. So fractMisMatch and
% fractMisMatch_trainingTrs will be very close.

% Compute it for all trials (not just the trials used for training, ie the
% equal number trials that were randomly selected).
% if pcaFlg 
% this doesn't work bc of nans in xx_s. For this reason if
% pcaFlg is 1, fractMisMatch will be quite different from
% fractMisMatch_trainingTrs bc the former is computed on data without
% dimension reduction (unlike how training was performed) but the later is with.
%     [label] = predict(SVMModel, xx_s); % predict(SVMModel, SVMModel.X);
%     label(isnan(sum(xx_s,2))) = NaN;
%     fractMisMatch(1) = sum(abs(yy - label)>0) / sum(~isnan(yy - label));
% else
    [label] = predict(SVMModel, xx); % predict(SVMModel, SVMModel.X);
    label(isnan(sum(xx,2))) = NaN;
    fractMisMatch_allTrs(1) = sum(abs(yy - label)>0) / sum(~isnan(yy - label));
% end


% Compute it for those exact trials used for training.
if pcaFlg
    [label] = predict(SVMModel, X_s); % predict(SVMModel, SVMModel.X);
    label(isnan(sum(X_s,2))) = NaN;    
else
    [label] = predict(SVMModel, X); % predict(SVMModel, SVMModel.X);
    label(isnan(sum(X,2))) = NaN;
end
fractMisMatch_trainingTrs(1) = sum(abs(Y - label)>0) / sum(~isnan(Y - label));
fprintf('%.3f = Fract classification error for the training dataset.\n', fractMisMatch_trainingTrs)


%%%%
% Estimate cross-validation predicted labels and scores.
% For every fold, kfoldPredict predicts class labels for in-fold
% observations using a model trained on out-of-fold observations.
CVSVMModel = crossval(SVMModel);
[elabel, escore] = kfoldPredict(CVSVMModel);

% Estimate the out-of-sample posterior probabilities
[ScoreCVSVMModel, ScoreParameters] = fitSVMPosterior(CVSVMModel);
[~, epostp] = kfoldPredict(ScoreCVSVMModel);

% How claassLoss is computed? I think: classLoss = 1 - mean(label == elabel)
classLossCV = kfoldLoss(CVSVMModel);
fprintf('%.3f = CV classification error (1 iteration)\n', classLossCV)
% a = diff([classLossCV, mean(label ~= elabel)]); % This is what you had,
% but I think it is wrong and you need to use the code below. Double check!
a = diff([classLossCV, mean(Y ~= elabel)]);
fprintf('%.3f = Difference in classLoss computed using kfoldLoss vs manually using kfoldPredict labels. This value is expected to be very small!\n', a)
if a > 1e-10
    a
    error('Why is there a mismatch? Read your comments above about "a".')
end


%% See how well the SVM decoder trained on our particular epoch can decode other time points.

% corrClass = NaN(size(traces_al_sm,1), size(traces_al_sm,3)); % frames x trials
corrClass = NaN(size(filtered,1), size(filtered,3)); % frames x trials

for itr = 1 : size(filtered, 3) % size(traces_al_sm, 3) % 
    
    % use non-smoother traces (remember in this case classification
    % accuracy on the window of training wont be very high, bc you trained
    % the classifier on the smoothed traces, but here you are predicting the
    % labels on the non-smoothed traces.)
% %     traces_bef_proj = traces_al_sm(:, ~NsExcluded, itr); % frames x neurons
%     traces_bef_proj = non_filtered(:,:,itr);

    % use the smoothed traces (smoothing window is of size ep)
% %     traces_bef_proj = filtered0(:, ~NsExcluded, itr); % frames x neurons 
    
    if pcaFlg
        traces_bef_proj = filtered_s(:, :, itr); % frames x neurons 
    else
        traces_bef_proj = filtered(:, :, itr); % frames x neurons 
    end
    
    if any(isnan(traces_bef_proj(:)))
        if ~all(isnan(traces_bef_proj(:))), error('how did it happen?'), end
    
    elseif ~isnan(Y(itr)) % ~isnan(choiceVec0(itr)) 
        l = predict(SVMModel, traces_bef_proj);
        corrClass(:, itr) = (l==Y(itr)); % (l==choiceVec0(itr)); % 
    end
end
% average performance (correct classification) across trials.
avePerf(:,1) = nanmean(corrClass, 2);  % frames x randomIters


%% 1) Set CV SVM models for a number of iterations. 2) Generate shuffled (chance) distributions for each iteration. 
% quality relative to shuffles

classLossTrain = NaN(1, numShuffs);
classLossTest = NaN(1, numShuffs);
classLossChanceTrain = NaN(1, numShuffs);
classLossChanceTest = NaN(1, numShuffs);

wNsHrLr_s = NaN(length(SVMModel.Beta), numShuffs);
biasHrLr_s = NaN(1, numShuffs);
wNsHrLrChance = NaN(length(SVMModel.Beta), numShuffs);
biasHrLrChance = NaN(1, numShuffs);

SVMModel_s_all = struct; % trained SVM models for all iterations.
SVMModelChance_all = struct; % trained SVM models for shuffled data for all iterations.
CVSVMModel_s_all = struct; % CV SVM models for all iterations.
CVSVMModelChance_all = struct; % CV SVM models for shuffled data for all iterations.

shflTrials_alls = NaN(length(Y), numShuffs);

for s = 1:numShuffs
    
    fprintf('Iteration %i...\n ----------------------- \n', s)
    
    
    if shuffleTrsForIters
        shflTrials = randperm(length(Y));
    else
        shflTrials = 1:length(Y);
    end
    X_s = X(shflTrials, :);
    Y_s = Y(shflTrials);
    shflTrials_alls(:,s) = shflTrials;
    
    %%%%%%%% reduce features by PCA
    if pcaFlg
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
    
    
    %% Actual data
    
    % trained dataset
    SVMModel_s = fitcsvm(X_s, Y_s, 'standardize', 1, 'ClassNames', cnam); % Linear Kernel
    classLossTrain(s) = mean(abs(Y_s - predict(SVMModel_s, X_s)));
    
    wNsHrLr_s(:, s) = SVMModel_s.Beta;
    biasHrLr_s(s) = SVMModel_s.Bias;

    % CV dataset
    CVSVMModel_s = crossval(SVMModel_s, 'kfold', 10); % CVSVMModel.Trained{1}: model 1 --> there will be KFold of these models. (by default KFold=10);    
    classLossTest(s) = kfoldLoss(CVSVMModel_s); % Classification loss (by default the fraction of misclassified data) for observations not used for training
        
    
    %% Shuffled data (for setting chance distributions: Y is shuffled so the link between trial and choice is gone).
    
    Y_s_shfld = Y_s(randperm(length(Y_s)));
    
    % shuffled trained dataset
    SVMModelChance = fitcsvm(X_s, Y_s_shfld, 'standardize', 1, 'ClassNames', cnam); %  % Linear Kernel
    classLossChanceTrain(s) = mean(abs(Y_s_shfld - predict(SVMModelChance, X_s)));

    wNsHrLrChance(:, s) = SVMModelChance.Beta;
    biasHrLrChance(:, s) = SVMModelChance.Bias;
    
    % shuffled CV dataset
    CVSVMModelChance = crossval(SVMModelChance, 'kfold', 10); % CVSVMModel.Trained{1}: model 1 --> there will be KFold of these models. (by default KFold=10);
    classLossChanceTest(s) = kfoldLoss(CVSVMModelChance); % Classification loss (by default the fraction of misclassified data) for observations not used for training

    
    %%    
    % Actual
    SVMModel_s_all(s).shuff = SVMModel_s; % actual data: trained SVM model
    SVMModelChance_all(s).shuff = SVMModelChance; % shuffled data: trained SVM model
    
    % Shuffled
    CVSVMModel_s_all(s).shuff = CVSVMModel_s; % actual data: CV SVM model
    CVSVMModelChance_all(s).shuff = CVSVMModelChance; % shuffled data: CV SCV model
    
    
end

classLoss = mean(classLossTest);

fprintf('%.3f = Average training classification error\n', mean(classLossTrain))
fprintf('%.3f = Average training classification error for shuffled data\n', mean(classLossChanceTrain))
fprintf('%.3f = Average cross-validated classification error\n', classLoss)
fprintf('%.3f = Average cross-validated classification error for shuffled data\n', mean(classLossChanceTest))


%% Compute and plot projections and classification accuracy (for each time point) for all CV models generated above.
% Remember CV progjections are not going to be very different from
% Training-data projections. This is because the decoder weights for CV SVM
% models are computed from the large percentage of data (90% if KFold is
% 10). As a result decoder weights should be very similar between CV and
% training datasets, hence the projections.

% What is really informative though is the classification accuracy, because
% that is computed on the 10% remaining test trials.

popClassifierSVM_CVdata_set_plot


%% Normalize weights (Beta) and in case weights were computed in multiple iterations (with different sets of trials), average beta across all iters (bagging : bootstrap aggregation)

% figure; imagesc(wNsHrLr)
% figure; errorbar(1:size(wNsHrLr,1), mean(wNsHrLr, 2), std(wNsHrLr, [], 2), 'k.')

% wNsHrLr_s
bLen = sqrt(sum(wNsHrLr.^2)); % norm of wNsHrLr for each rand
% figure; plot(bLen)
wNsHrLrNorm = bsxfun(@rdivide, wNsHrLr, bLen); % normalize b of each rand by its vector length
wNsHrLrAve = mean(wNsHrLrNorm, 2); % average of normalized b across all rands.
wNsHrLrAve = wNsHrLrAve / norm(wNsHrLrAve); % normalize it so the final average vector has norm of 1.
% figure; plot(bNsHrLrAve)
% figure; errorbar(1:size(wNsHrLr,1), mean(wNsHrLrNorm, 2), std(wNsHrLrNorm, [], 2), 'k.')


%% Compare fraction of mismatch in classification between original weights and normalized (and bagged) weights.
% wNsHrLrAve = wNsHrLr;

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

fractMisMatch_normalized_iterAveraged_beta = sum(abs(yy - label)>0) / sum(~isnan(yy - label));

fractMisMatchAllTrs_originalBeta_vs_normBaggedBeta = [nanmean(fractMisMatch_allTrs) fractMisMatch_normalized_iterAveraged_beta]


%{
% compare with fractMisMatch on each iter ... see if doing several
% iters helped w better prediction:
figure; hold on,
plot([0 length(fractMisMatch_allTrs)], [fractMisMatch_normalized_iterAveraged_beta fractMisMatch_normalized_iterAveraged_beta])
plot(fractMisMatch_allTrs)
title(sprintf('%.3f  %.3f', nanmean(fractMisMatch_allTrs), fractMisMatch_normalized_iterAveraged_beta))
%}

%{
%% plot bias
figure('name', 'bias term'); subplot(211), plot(biasHrLr)
subplot(212),  errorbar( mean(biasHrLr), std(biasHrLr), 'k.')
%}


%% Main summary plots

% plots of projections, and classification accuracy for all trials (the
% vast majority of them are the training dataset).
popClassifierSVM_plots



%% Additional plots

if plot_rand_choicePref
    % Compare svm weights with random weights
    popClassifierSVM_rand

    % Compare SVM weights with ROC choicePref
    popClassifierSVM_choicePref
end

