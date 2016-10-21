mouse = 'fni17';
imagingFolder = '151102'; %'151029'; %  '150916'; % '151021';
mdfFileNumber = [1,2];  % 3; %1; % or tif major

% the following are needed for setting stim-aligned traces.
ep_ms = [1000 1300]; %[700 900]; % make sure no trial includes go tone before the end of ep % ep_ms will be used for setting X (trials x neurons, average of neural activity in window ep relative to stimulus onset).
th_stim_dur = 800; % min stim duration to include a trial in timeStimOnset
% the following are needed for setting X, Y, TrsExcluded, NsExcluded
trialHistAnalysis = 0;
outcome2ana = 'corr'; % only used if trialHistAnalysis is 0; % '', corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials)
strength2ana = 'all'; % only used if trialHistAnalysis is 0; % 'all', easy', 'medium', 'hard' % What stim strength to use for training?
thAct = 5e-4; % 1e-4; % quantile(spikeAveEpAveTrs, .1);
thTrsWithSpike = 1; % 3; % 1; % ceil(thMinFractTrs * size(spikeAveEp0,1)); % 30  % remove neurons with activity in <thSpTr trials.


%% Load stimrate and set the following variables

% traces_al_stim: frames x units x trials; stimulus aligned traces. Only for active neurons and valid (non-nan) trials.
% time_aligned_stim: 1 x frames; time points for non-filtered trace.
% X: trials x neurons; includes average of window ep (defined in SVM codes) for each neuron at each trial. 
% Y: trials x 1; animal's choice on the current trial (0: LR, 1:HR)

% stimrate: trials x 1; stimulus rate of each trial.


signalCh = 2;
pnev2load = []; %7 %4 % what pnev file to load (index based on sort from the latest pnev vile). Set [] to load the latest one.
postNProvided = 1; % whether the directory cotains the postFile or not. if it does, then mat file names will be set using postFile, otherwise using pnevFile.
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load, postNProvided);
[pd, pnev_n] = fileparts(pnevFileName);
postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));

load(postName, 'stimAl_allTrs', 'stimrate', 'time1stSideTry', 'timeCommitCL_CR_Gotone', 'timeStimOnset', 'timeStimOffset', 'outcomes', 'allResp_HR_LR', 'cb')
% load('SVM_151029_003_ch2-PnevPanResults-160426-191859.mat')


% Set traces_al_stim: same as traces_al_stimAll except that some trials are
% set to nan: bc their stim duration is < th_stim_dur ms or bc their go
% tone happens before ep(end) ms. (In traces_al_stimAll, all trials are
% included).
traces_al_stimAll = stimAl_allTrs.traces;
time_aligned_stim = stimAl_allTrs.time;
eventI = stimAl_allTrs.eventI;

popClassifier_setToRmv % set toRmv, ie trials in which go tone happened before ep_ms(end) and trials that have <th_stim_dur duration. 

traces_al_stim = traces_al_stimAll;
traces_al_stim(:,:,toRmv) = nan;


% Set X, Y, TrsExcluded, NsExcluded
popClassifier_setXY % set X, Y, TrsExcluded, NsExcluded

% Exclude nan trials and non-active neurons from traces_al_stim
traces_al_stim = traces_al_stim(:, ~NsExcluded, ~trsExcluded);

% Exclude nan trials from stimrate
stimrate = stimrate(~trsExcluded);


%%
dataTensor = traces_al_stim; % non_filtered; % stimulus aligned traces. Only for active neurons and valid (non-nan) trials.
all_times = time_aligned_stim; % time_aligned;
% dataTensor = traces_al_1stSideTry;
% all_times = time_aligned_1stSideTry;

[T, N, R] = size(dataTensor);

%% average across multiple times (downsample dataTensor; not a moving average. we only average every regressBins points.)
regressBins = 2;
dataTensor = dataTensor(1:regressBins*floor(T/regressBins), : , :);
dataTensor = squeeze(mean(reshape(dataTensor, regressBins, floor(T/regressBins), N, R), 1)); % not a moving average. we only average every regressBins points. 
all_times = all_times(1:floor(T/regressBins)*regressBins);
all_times = round(mean(reshape(all_times, regressBins, floor(T/regressBins)),1), 2);

%% preprocess dataTensor: do mean subtraction and feature normalization (for each neuron)
[T, N, R] = size(dataTensor);
meanN = mean(reshape(permute(dataTensor, [1 3 2]), T*R, N)); % 1xneurons; includes the average of all time points for all trials for each neuron.
stdN = std(reshape(permute(dataTensor, [1 3 2]), T*R, N));
meanN = mean(X); % X: trials x neurons; includes average of window ep (defined in SVM codes) for each neuron at each trial.
stdN = std(X);
dataTensor = bsxfun(@times, bsxfun(@minus, dataTensor, meanN), 1./(stdN+sqrt(0))); 

%% plot average of dataTensor per decision (average across trials and neurons)
figure;
hold on
plot(all_times, mean(mean(dataTensor(:, :, Y==0), 3), 2), 'b')
plot(all_times, mean(mean(dataTensor(:, :, Y==1), 3), 2), 'r')
plot(all_times, mean(mean(dataTensor(:, :, :), 3), 2), 'k')
legend('low rate', 'high rate', 'all')
xlabel('time (ms)') % since stimulus onset.
ylabel('normalized firing rates')

%% alignment of top variance subspaces 
numDim = 10; % define dimensionality of subspace
[PCs_t, Summary] = pca_t(dataTensor, numDim); % identify subspaces (FN: each row is the PC space (neurons x numPCs) for a particular time point).
aIx = nan(T, T); % alignement index between subspaces
for i = 1:T
    for j = 1:T
        aIx(i,j) = alignIx(squeeze(PCs_t(i, :, :)), squeeze(PCs_t(j, :, :))); % compute alignment between PC spaces at time points i and j. % FN: this is done by computing variance explained after projecting pc_i onto pc_j divided by variance explained before the projection.
    end
end
figure;
imagesc(all_times, all_times, aIx);
colormap(colormap('jet'))
title('alignment index')
colorbar
caxis([0 1])
axis square
xlabel('time (ms)')
ylabel('time (ms)')

%% TDR analysis: FN: 1st we do denoising. Then for each neuron and at each timepoint, we fit a regression model that predicts neural response given stimulus and choice.
stim = stimrate(:);
decision = Y;
cb = 16;
kfold = 10;
codedParams = [[stim(:)-cb]/range(stim(:)) [decision(:)-mean(decision(:))]/range(decision(:)) ones(R, 1)]; % trials x 1: stimulusRate_choice_ones; stimulusRate and choice are mean subtracted.
[~, codedParams] = normVects(codedParams); % each parameter is unit vector % normalize each column to have length 1.
% cb = 16; stimrate_norm = ((stimrate - cb)/max(abs(stimrate(:) - cb)));
numPCs = 10;
[dRAs, normdRAs, Summary] = runTDR(dataTensor, numPCs, codedParams, [], kfold); 
% dRAs: times(frames) x neurons x predictors; normalized beta coefficients (ie coefficients of stimulus, choice, and offset for each neuron at each timepoint)


% Now we compute the angle (times x times) between 2 subspaces of neurons
% that represent particular features (stimulus,choice,offset) at all timepoints.
%
% angle(t1,t2,4) = Angle between 2 vectors of size 1 x neurons. 1 vector
% represents stimulus beta at time t1 (dRAs(t1,:,1)) and the other
% represents choice beta at time t2 dRAs(t2,:,1).
angle = real(acos(abs(dRAs(:,:,1)*dRAs(:,:,1)')))*180/pi; % times x times
angle(:,:,2) = real(acos(abs(dRAs(:,:,2)*dRAs(:,:,2)')))*180/pi;
angle(:,:,3) = real(acos(abs(dRAs(:,:,3)*dRAs(:,:,3)')))*180/pi;
angle(:,:,4) = real(acos(abs(dRAs(:,:,1)*dRAs(:,:,2)')))*180/pi;

%%
figure;
subplot(221)
plot(all_times, Summary.R2_t, 'k')
xlabel('time (ms)')
ylabel('R^2')

subplot(222);
hold on
plot(all_times, Summary.R2_tk(:,1), 'g')
plot(all_times, Summary.R2_tk(:,2), 'b')
plot(all_times, Summary.R2_tk(:,3), 'c')
xlabel('time (ms)')
ylabel('R2 of each predictor') % signal contribution to firing rate
legend('stimulus', 'decision', 'offset')

subplot(223); hold on; 
a = mean(dRAs(:,:,1),2);
plot(all_times, a, 'g')
a = mean(dRAs(:,:,2),2);
plot(all_times, a, 'b')
a = mean(dRAs(:,:,3),2);
plot(all_times, a, 'c')
ylabel({'Normalized beta', '(average across neurons)'})
legend('stimulus', 'decision', 'offset')



figure;
subplot(221)
imagesc(all_times, all_times, angle(:,:,1));
colormap(flipud(colormap('jet')))
colorbar
title('angle stim')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(222)
imagesc(all_times, all_times, angle(:,:,2));
colormap(flipud(colormap('jet')))
colorbar
title('angle decision')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(223)
imagesc(all_times, all_times, angle(:,:,3));
colormap(flipud(colormap('jet')))
colorbar
title('angle offset')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(224)
imagesc(all_times, all_times, angle(:,:,4));
colormap(flipud(colormap('jet')))
colorbar
title('angle stim vs decision')
caxis([0 90])
xlabel('decision time (ms)')
ylabel('stimulus time (ms)')
axis square

%%
sRA = optimize_oTDR(dataTensor, codedParams, [], []);

%%
dataTensor_proj(:, 1, :) = projectTensor(dataTensor, squeeze(sRA(:, 1)));
dataTensor_proj(:, 2, :) = projectTensor(dataTensor, squeeze(sRA(:, 2)));

uniqueStim = unique(stim);
uniqueDecision = unique(decision);
S = length(uniqueStim);
D = length(uniqueDecision);
projStim = [];
for s = 1:S
    for d = 1:D
        msk = (stim == uniqueStim(s)) & (decision == uniqueDecision(d));
        proj1(:, s, d) = mean(squeeze(dataTensor_proj(: ,1, msk)), 2);
        proj2(:, s, d) = mean(squeeze(dataTensor_proj(: ,2, msk)), 2);        
    end
end

clr = redgreencmap(S, 'interpolation', 'linear');
figure;
l = {};
subplot(121)
hold on
for s = 1:S
    plot(all_times, proj1(:, s, 1), '--', 'color', clr(s, :));
    h(s) = plot(all_times, proj1(:, s, 2), '-', 'color', clr(s, :));
    l{s} = [num2str(uniqueStim(s)) 'Hz']; 
end
title('stimulus projection')
legend(h, l)
 
subplot(122)
hold on
for s = 1:S
    plot(all_times, proj2(:, s, 1), '--', 'color', clr(s, :));
    h(s) = plot(all_times, proj2(:, s, 2), '-', 'color', clr(s, :));
end
title('choice projection')
legend(h, l)

%%
