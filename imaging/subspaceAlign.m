imagingFolder = '151029';
mouse = 'fni17';
mdfFileNumber = 3;
setInhibitExcit = true;
rmv_timeGoTone_if_stimOffset_aft_goTone = 0;
rmv_time1stSide_if_stimOffset_aft_1stSide = 0;
plot_ave_noTrGroup = 1; % Plot average traces across all neurons and all trials aligned on particular trial events.
frameLength = 1000/30.9; % sec.

[alldata, alldataSpikesGood, alldataDfofGood, goodinds, good_excit, good_inhibit, outcomes, allResp, allResp_HR_LR, ...
        trs2rmv, stimdur, stimrate, stimtype, cb, timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, ...
        timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks, imfilename] = ....
    imaging_prep_analysis(mouse, imagingFolder, mdfFileNumber, setInhibitExcit, ...
        rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, plot_ave_noTrGroup, frameLength);

    
    %%
    
   
    neuronType = 2; % 0: excitatory, 1: inhibitory, 2: all types.
trialHistAnalysis = 0; % more parameters are specified in popClassifier_trialHistory.m
    iTiFlg = 1; % 0: short ITI, 1: long ITI, 2: all ITIs.
    prevSuccessFlg = false; % true previous sucess trials; false: previous failure.
    vec_iti = [0 9 30]; % [0 10 30]; %[0 6 9 12 30]; % [0 7 30]; % [0 10 30]; % [0 6 9 12 30]; % use [0 40]; if you want to have a single iti bin and in conventioinal analysis look at the effect of current rate on outcome.
    
numShuffs = 10; % 100 % number of iterations for getting CV models and shuffled data.
onlyCorrect = 1; % If 1, analyze only correct trials.

% Set nPre and nPost to nan to make sure frames before and after
% alignedEvent don't have any other events.
% Set to [] to include all exisiting frames before and after the
% alignedEvent (regardless of whether they include other frames or not).
% Set to any other value if you wan to manually specify the number of frames
% before and after alignedEvent in the aligned traces (ie traces_al_sm).


% alignedEvent: 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward', 'commitIncorrResp'
if trialHistAnalysis
    alignedEvent = 'initTone';
else
    alignedEvent = 'goTone'; 'stimOn';'commitIncorrResp'; 'reward'; 'goTone';  % 'stimOn'; % what event align traces on. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward'
end

smoothedTraces = 0; % if 1, projections and class accuracy of temporal traces will be computed on smoothed traces (window size same as ep, ie training window size).


%% all data
% remember traces_al_sm has nan for trs2rmv as well as trs in alignedEvent that are nan.
traces = alldataSpikesGood; % alldataSpikesGoodExc; % alldataSpikesGoodInh; % alldataSpikesGood;  % traces to be aligned.
% alignedEvent = 'stimOn'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward'
dofilter = true; % true;
traceTimeVec = {alldata.frameTimes};

% % set nPre and nPost to nan if you want to go with the numbers that are based on eventBef and eventAft.
% % set to [] to include all frames before and after the alignedEvent.
nPreFrames = [];
nPostFrames = [];
% alignedEvent = 'reward';    % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward', 'commitIncorrResp'
[traces_al_sm, time_aligned, eventI] = alignTraces_prePost_filt(traces, traceTimeVec, alignedEvent, frameLength, dofilter, ...
    timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, nPreFrames, nPostFrames);


a = find(sum(sum(~isnan(traces_al_sm),1),3), 1);
allTrs2rmv = find(squeeze(sum(isnan(traces_al_sm(:,a,:)))));
outcomes(allTrs2rmv) = NaN;
allResp_HR_LR(allTrs2rmv) = NaN;
choiceVec0 = allResp_HR_LR';  % trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.
choiceVec0(outcomes~=1) = NaN; % analyze only correct trials.

X = traces_al_sm(:, :, ~isnan(choiceVec0)); % call this all response at the whole trial
mskNeus = find(squeeze(sum(sum(~isnan(X),1), 3)));
X = X(:,mskNeus, :);
%% extract stimulus epoch and decision epoch

% remember traces_al_sm has nan for trs2rmv as well as trs in alignedEvent that are nan.

traces = alldataSpikesGood; % alldataSpikesGoodExc; % alldataSpikesGoodInh; % alldataSpikesGood;  % traces to be aligned.
% alignedEvent = 'stimOn'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward'
dofilter = true; % true;
traceTimeVec = {alldata.frameTimes};

% % set nPre and nPost to nan if you want to go with the numbers that are based on eventBef and eventAft.
% % set to [] to include all frames before and after the alignedEvent.
nPreFrames = nan;
nPostFrames = nan;
% alignedEvent = 'reward';    % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward', 'commitIncorrResp'
[traces_al_sm, time_aligned, eventI] = alignTraces_prePost_filt(traces, traceTimeVec, alignedEvent, frameLength, dofilter, ...
    timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, nPreFrames, nPostFrames);
a = find(sum(sum(~isnan(traces_al_sm),1),3), 1);
allTrs2rmv = find(squeeze(sum(isnan(traces_al_sm(:,a,:)))));

X1 = traces_al_sm(time_aligned<0, mskNeus, ~isnan(choiceVec0)); % call this stimulus response
X2 = traces_al_sm(time_aligned>0, mskNeus, ~isnan(choiceVec0)); % call this decision response
X1 = X1(1:end-4, :, :);
X2 = X2(2:end, :, :);

%%

% % % X1N = reshape(permute(X1, [1 3 2]), size(X1,1)*size(X1,3), size(X1,2));
% % % X2N = reshape(permute(X2, [1 3 2]), size(X2,1)*size(X2,3), size(X2,2));
% % % XN = reshape(permute(X, [1 3 2]), size(X,1)*size(X,3), size(X,2));


X1N = squeeze(mean(X1, 1))';
X2N = squeeze(mean(X2, 1))';
XN = squeeze(mean(X, 1))';

mu = mean(XN);
sd = std(XN);

XN = bsxfun(@times, bsxfun(@minus, XN, mu), 1./sd);
X1N = bsxfun(@times, bsxfun(@minus, X1N, mu), 1./sd);
X2N = bsxfun(@times, bsxfun(@minus, X2N, mu), 1./sd);

dim = 10;
numSamples = 1000;
Ix = nan(numSamples, 1);
trials_i = 1:size(X1, 3);
for i = 1:numSamples
    trials_i = randi(size(X1, 3), size(X1, 3), 1);
    trials_i1 = repmat(trials_i(:).', size(X1,1), 1);
    trials_i2 = repmat(trials_i(:).', size(X2,1), 1);
    X1N_i = X1N(trials_i(:), :);
    X2N_i = X2N(trials_i(:), :);
    X1N_i = bsxfun(@minus, X1N_i, mean(X1N_i));
    X2N_i = bsxfun(@minus, X2N_i, mean(X2N_i));
    [Q1, ~, ~] = pca(X1N_i); 
    [Q2, ~, ~] = pca(X2N_i); 

    Ix(i) = alignIx(X1N_i.', Q2(:,1:dim));
end

randIx = sampleRandSubspaces(dim, cov(XN), 'alignIx', numSamples);

[~, pVal] = ttest2(Ix, randIx, 'tail', 'left','varType', 'unequal');
hc = hist(randIx,0:0.01:1);
hd = hist(Ix, 0:0.01:1);

figure;
hold on
bar(0:0.01:1, hc, 'facecolor', [.7 .7 .7], 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
bar(0:0.01:1, hd, 'facecolor', 'g', 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
plot(mean(randIx), 0,'ko', 'markerfacecolor', [.7 .7 .7]);
plot(mean(Ix), 0,'ko', 'markerfacecolor', 'g');
title(['P-value ' num2str(pVal)]);
legend('chance distribution', 'data')
xlabel('alignment index')
ylabel('count')
xlim([0 1])
set(gca, 'FontSize', 16)

%% averaged decision
XN = [];
Y = choiceVec0(~isnan(choiceVec0))>0;
XN(:, :, 1) = mean(X(:, :, Y), 3);
XN(:, :, 2) = mean(X(:, :, ~Y), 3);
XN = reshape(permute(XN, [1 3 2]), size(X,1)*2, size(X,2));
mu = mean(XN);
sd = std(XN);
XN = bsxfun(@times, bsxfun(@minus, XN, mu), 1./sd);

X1N = [];
X1N(:, :, 1) = mean(X1(:, :, Y), 3);
X1N(:, :, 2) = mean(X1(:, :, ~Y), 3);
X1N = reshape(permute(X1N, [1 3 2]), size(X1,1)*2, size(X1,2));
X1N = bsxfun(@times, bsxfun(@minus, X1N, mu), 1./sd);

X2N = [];
X2N(:, :, 1) = mean(X2(:, :, Y), 3);
X2N(:, :, 2) = mean(X2(:, :, ~Y), 3);
X2N = reshape(permute(X2N, [1 3 2]), size(X2,1)*2, size(X2,2));
X2N = bsxfun(@times, bsxfun(@minus, X2N, mu), 1./sd);



dim = 10;
numSamples = 1000;
X1N_i = X1N;
X2N_i = X2N;
X1N_i = bsxfun(@minus, X1N_i, mean(X1N_i));
X2N_i = bsxfun(@minus, X2N_i, mean(X2N_i));
[Q1, ~, ~] = pca(X1N_i); 
[Q2, ~, ~] = pca(X2N_i); 

Ix = alignIx(X1N_i.', Q2(:,1:dim));


randIx = sampleRandSubspaces(dim, cov(XN), 'alignIx', numSamples);
pVal = sum(Ix>=randIx)./numSamples;
figure;
hold on
hist(randIx, 0:0.01:1);
h = findobj(gca,'Type','patch');
h.FaceColor = [0.5 0.5 0.5];
h.EdgeColor = [0.5 0.5 0.5];
plot(Ix, 0, 'go', 'markerfacecolor', 'r');
title(['P-value ' num2str(pVal)]);
legend('chance distribution', 'data')
xlabel('alignment index')
ylabel('count')
xlim([0 1])
% % 
% % 
% % Ix = nan(numSamples, 1);
% % for i = 1:numSamples
% %     times_i1 = randi(size(X1, 1), size(X1, 1), 1);
% %     times_i2 = randi(size(X2, 1), size(X2, 1), 1);
% %     times_i1 = repmat(times_i1(:).', 2, 1);
% %     times_i2 = repmat(times_i2(:).', 2, 1);
% %     
% %     X1N_i = X1N(times_i1(:), :);
% %     X2N_i = X2N(times_i2(:), :);
% %     X1N_i = bsxfun(@minus, X1N_i, mean(X1N_i));
% %     X2N_i = bsxfun(@minus, X2N_i, mean(X2N_i));
% %     [Q1, ~, ~] = pca(X1N_i); 
% %     [Q2, ~, ~] = pca(X2N_i); 
% % 
% %     Ix(i) = alignIx(X1N_i.', Q2(:,1:dim));
% % end
% % randIx = sampleRandSubspaces(dim, cov(XN), 'alignIx', numSamples);
% % 
% % [~, pVal] = ttest2(Ix, randIx, 'tail', 'left','varType', 'unequal');
% % hc = hist(randIx,0:0.01:1);
% % hd = hist(Ix, 0:0.01:1);
% % 
% % figure;
% % hold on
% % bar(0:0.01:1, hc, 'facecolor', [.7 .7 .7], 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
% % bar(0:0.01:1, hd, 'facecolor', 'g', 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
% % plot(mean(randIx), 0,'ko', 'markerfacecolor', [.7 .7 .7]);
% % plot(mean(Ix), 0,'ko', 'markerfacecolor', 'g');
% % title(['P-value ' num2str(pVal)]);
% % legend('chance distribution', 'data')
% % xlabel('alignment index')
% % ylabel('count')
% % xlim([0 1])
% % set(gca, 'FontSize', 16)
