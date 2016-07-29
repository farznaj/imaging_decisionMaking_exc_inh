function [alldata, alldataSpikesGood, alldataDfofGood, goodinds, good_excit, good_inhibit, outcomes, allResp, allResp_HR_LR, ...
        trs2rmv, stimdur, stimrate, stimtype, cb, timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, ...
        timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks, imfilename, pnevFileName] = ....
    imaging_prep_analysis(mouse, imagingFolder, mdfFileNumber, setInhibitExcit, ...
        rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, plot_ave_noTrGroup, frameLength);
%
% This is the main and starting function for the analysis of your imaging
% data. It gives you the vars that you need for further analyses.
% it used to be named aveTrialAlign_setVars
%
% pnev_manual_comp_setVars is also a very nice script (together with
% pnev_manual_comp_match) that allows you to plot and compare the trace and
% ROIs of 2 different methods.
%  you can use it to compare Eftychios vs manual. Or 2 different channels.
% Or 2 different methods of Eftychios, etc.
%
% It seems the following is not a concern when using Eftychios's algorithm:
% worry about bleedthrough, and visual artifact.
%
% outcomes:
%    1: success, 0: failure, -1: early decision, -2: no decision, -3: wrong initiation,
%   -4: no center commit, -5: no side commit
%
% Farzaneh Najafi (2016)
%
% Example input variales:
%{
mouse = 'fni17';
imagingFolder = '151029'; % '151102'; % '150916'; % '151021';
mdfFileNumber = 3; % 1; % or tif major

rmv_timeGoTone_if_stimOffset_aft_goTone = 0; % if 1, trials with stimOffset after goTone will be removed from timeGoTone (ie any analyses that aligns trials on the go tone)
rmv_time1stSide_if_stimOffset_aft_1stSide = 0; % if 1, trials with stimOffset after 1stSideTry will be removed from time1stSideTry (ie any analyses that aligns trials on the 1stSideTry)

plot_ave_noTrGroup = 0; % Set to 1 when analyzing a session for the 1st time. Plots average imaging traces across all neurons and all trials aligned on particular trial events. Also plots average lick traces aligned on trial events.

setInhibitExcit = 0; %true; % if 1, inhibitory and excitatory neurons will be identified unless inhibitRois is already saved in imfilename (in which case it will be loaded).

frameLength = 1000/30.9; % sec.
%}

home

%% Good days
% imagingFolder = '151029'; % '151021';
% mdfFileNumber = 3;


%% Set some initial variables.

    assessInhibitClass = 0; % if 1 and inhibitRois not already saved, you will evaluate identification of inhibitory neurons (ie if sigTh is doing a good job).
    saveInhibitRois = 0; % if 1 and inhibitRois not already saved, ROIs that were identified as inhibit will be saved in imfilename. (ie the output of inhibit_excit_setVars will be saved).
    sigTh = 1.13; % signal to noise threshold for identifying inhibitory neurons on tdtomato channel. eg. sigTh = 1.2;
    % 1.1: excit is safe, but u should check inhibit with low sig/surr to make sure they are not excit.
    % 1.2: u are perhaps missing some inhibit neurons.
    % 1.13 can be good too.
    
signalCh = 2;
pnev2load = []; %7 %4 % what pnev file to load (index based on sort from the latest pnev vile). Set [] to load the latest one.

autoTraceQual = 0; %1; % if 1, automatic measure for trace quality will be used.
normalizeSpikes = 1; % if 1, spikes trace of each neuron will be normalized by its max.


excludeShortWaitDur = true; % waitdur_th = .032; % sec  % trials w waitdur less than this will be excluded.
excludeExtraStim = false;
allowCorrectResp = 'change'; % 'change'; 'remove'; 'nothing'; % if 'change': on trials that mouse corrected his choice, go with the original response.
uncommittedResp = 'nothing'; % 'change'; 'remove'; 'nothing'; % what to do on trials that mouse made a response (licked the side port) but did not lick again to commit it.

thbeg = 5; % n initial trials to exclude.

evaluateEftyAC = 0; % A and C for each component will be plotted 1 by 1 for evaluation of of Efty's results. 
furtherAnalyses = 0; % analyses related to choicePref and SVM will be performed.
plot_ave_trGroup = false; % Plot average traces across all neurons for different trial groups aligned on particular trial events.
plotTrialTraces1by1 = false; % plot traces per neuron and per trial showing all trial events
compareManual = false; % compare results with manual ROI extraction

setNaN_goToneEarlierThanStimOffset = 0; % if 1, set to nan eventTimes of trials that had go tone earlier than stim offset... if 0, only goTone time will be set to nan.

manualExamineTraceQual = 0; % if 0, traceQuality array needs to be saved.
    saveTraceQual = 0; % it will only take effect if manualExamineTraceQual is 1.
    analyzeQuality = [1 2]; % 1(good) 2(ok-good) 3(ok-bad) 4(bad) % trace qualities that will be analyzed. It will only take effect if manualExamineTraceQual is 1.
orderTraces = 0; % if 1, traces will be ordered based on the measure of quality from high to low quality.
%{
helpedInit = []; % [100];
helpedChoice = []; %31;
defaultHelpedTrs = 0; % if 1, the program assumes that no trial was helped.
saveHelpedTrs = 0; % it will only take effect if defaultHelpedTrs is false. If 1, helpedTr fields will be added to alldata.
%}
% analyzeOutcomes = {'all'}; % {'success', 'failure'}; % outcomes that will be analyzed.
% outName = 'fni17-151016';


%%

% set the names of imaging-related .mat file names.
% remember the last saved pnev mat file will be the pnevFileName
% signalCh = 2; % because you get A from channel 2, I think this should be always 2.
% pnev2load = [];
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[~,f] = fileparts(pnevFileName);
disp(f)
cd(fileparts(imfilename))


% load alldata

% load(imfilename, 'all_data'), all_data = all_data(1:end-1);   %from the imaging file.  % alldata = removeBegEndTrs(alldata, thbeg);
% use the following if you want to go with the alldata of the behavior folder
% set filenames
[alldata_fileNames, ~] = setBehavFileNames(mouse, {datestr(datenum(imagingFolder, 'yymmdd'))});
% sort it
[~,fn] = fileparts(alldata_fileNames{1});
a = alldata_fileNames(cellfun(@(x)~isempty(x),cellfun(@(x)strfind(x, fn(1:end-4)), alldata_fileNames, 'uniformoutput', 0)))';
[~, isf] = sort(cellfun(@(x)x(end-25:end), a, 'uniformoutput', 0));
alldata_fileNames = alldata_fileNames(isf);
% load the one corresponding to mdffilenumber.
[all_data, ~] = loadBehavData(alldata_fileNames(mdfFileNumber)); % , defaultHelpedTrs, saveHelpedTrs); % it removes the last trial too.
fprintf('Total number of behavioral trials: %d\n', length(all_data))


% begTrs = [0 cumsum(trials_per_session)]+1;
% begTrs = begTrs(1:end-1);
begTrs = 1; % 1st trial of each session


%% Some plots and prints of info related to behavior and imaging of behavior

%{
* stimulus is played for stimDuration which equal stimDur_diff (or if it is 0, waitDur) + extrastim_dur + stimdur_aftrew .
* total waitdur (ie since stim onset, when mouse was allowed to do cent commit) equals waitDuration + postStimDelay
%}


% use the following plot to decide on thbeg: number of begining trials that
% you want to exclude.
% stimDur_diff
figure; hold on
title(sprintf('thbeg = %d', thbeg))
plot([all_data.waitDuration]+[all_data.postStimDelay])
plot([all_data.stimDuration])
plot([all_data.extraStimDuration])
plot([all_data.stimDur_aftRew])
plot([all_data.stimDur_diff])
% plot([thbeg thbeg],[-1 12], 'k')
legend('waitDur', 'stimDuration', 'extraStimDur', 'stimDur\_aftRew', 'stimDur\_diff')
set(gcf, 'name', 'stimDuration= stimDur_diff (or if it is 0, waitDur) + extrastim_dur + stimdur_aftrew .')
m = max([all_data.stimDuration]);
ylim([-1 m+1])
set(gca,'tickdir','out')


% u = unique({all_data.rewardStage}); disp(u);
rs = {all_data.rewardStage}; 
ave_rewardStage_allowCorr_chooseSide = [nanmean(strcmp(rs, 'Allow correction')), nanmean(strcmp(rs, 'Choose side'))];
fprintf('Fraction of trials (rewardStage): allowCorr= %.3f. chooseSide= %.3f\n', ave_rewardStage_allowCorr_chooseSide)
% rsi = zeros(1, length(rs)); 
% rsi(strcmp(rs, 'Choose side')) = 1; 
% figure; plot(rsi); ylim([-.1 1.1])
% xlabel('Trials'), ylabel('chooseSide'), box off, set(gca,'tickdir','out')



load(imfilename, 'framesPerTrial', 'trialNumbers', 'frame1RelToStartOff', 'trialCodeMissing')
trialNumbers(isnan(framesPerTrial)) = [];
frame1RelToStartOff(isnan(framesPerTrial)) = [];
framesPerTrial(isnan(framesPerTrial)) = [];

if ~exist('trialCodeMissing', 'var') % for those days that you didn't save this var.
    trialCodeMissing = [];
end

fprintf('Total number of imaged trials: %d\n', length(trialNumbers))
if ~any(trialCodeMissing)
    cprintf('blue', 'All trials are triggered in MScan :)\n')
else
    cprintf('blue', 'There are non-triggered trials in MScan! Trial %d\n', find(trialCodeMissing))
end



%%
%%%%%%%%%%%%%%%%%%%%%%%%% Load neural data, merge them into all_data, and assess trace quality %%%%%%%%%%%%%%%%%%%%%%%

%% Load vars related to manual method

if compareManual
    
    load(imfilename, 'imHeight', 'imWidth', 'pmtOffFrames')
    load(pnevFileName, 'activity_man_eftMask*')
    %     load(imfilename, 'activity_man_eftMask')
    
    %{
    load(imfilename, 'rois', 'activity') % , 'imHeight', 'imWidth', 'pmtOffFrames')
    load(pnevFileName, 'A') % load('demo_results_fni17-151102_001.mat', 'A2')
    spatialComp = A; % A2 % obj.A;
    clear A
    %}
    
    %% manual activity and dfof
    activity_man = activity_man_eftMask_ch2;
    
    %     activity_man = activity;
    clear activity activity_man_eftMask*
    
    % Compute df/f for the manually found activity trace.
    gcampCh = 2; smoothPts = 6; minPts = 7000; %800;
    dFOF_man = konnerthDeltaFOverF(activity_man, pmtOffFrames{gcampCh}, smoothPts, minPts);
    
    
    %% masks and matching ROIs between the 2 methods.
    %     csfrs = [0 cumsum(framesPerTr)];
    
    % if you computed manual activity on eft mask:
    matchedROI_idx = 1:size(dFOF_man,2); % there is a 1-1 match between ROIs
    
    % set contours for Eft ROIs
    %{
    contour_threshold = .95;
    [CCorig, CR, COMs] = ROIContoursPnev(spatialComp, imHeight, imWidth, contour_threshold); % P.d1, P.d2
    CC = ROIContoursPnev_cleanCC(CCorig);
    % set masks for Eft ROIs
    mask_eft = maskSet(CC, imHeight, imWidth);
    
    
    % set masks for manual ROIs
    CC_rois = cell(1, length(rois)); % Set the coordinates for the manual method in [y x] format for each ROI. (similar to CC of Eftychios method.)
    for rr = 1:length(rois)
        CC_rois{rr} = [rois{rr}.mnCoordinates(:, 2)'; rois{rr}.mnCoordinates(:, 1)'];
    end
    mask_manual = maskSet(CC_rois, imHeight, imWidth);
    
    % find manual ROIs that match Eft ROIs
    refMask = mask_eft;
    toMatchMask = mask_manual;
    matchedROI_idx = matchROIs_sumMask(refMask, toMatchMask); % matchedROI_idx(i)=j means ROI i of refMask matched ROI j of toMatchMask.
    %}
    
end


%% Assess pixel shifts (ie output of motion correction), pmtOffFrames and badFrames (frames with too much motion)

load(imfilename, 'outputsDFT')
figure; plot(outputsDFT{1}(:,2:3))
legend('row shift', 'column shift')
xlabel('Frame')
ylabel('Pixel shift')


load(imfilename, 'pmtOffFrames')
pof = cellfun(@sum, pmtOffFrames);
if any(pof)
    cprintf('red', 'Number of pmtOffFrames on each channel = %d %d\n', pof)
    warning('Take care of pmtOffFrames!')
end


load(imfilename, 'badFrames')
bf = cellfun(@sum, badFrames);
if any(bf)
    cprintf('red', 'Number of badFrames on each channel = %d %d\n', bf)
    warning('Take care of badFrames!')
end


%% Load and clean Efty's vars

fprintf('Loading Eftys vars...')
load(pnevFileName, 'C', 'C_df', 'S', 'A', 'P', 'f')
fprintf('...done\n')
load(imfilename, 'Nnan_nanBeg_nanEnd')
% normalizeSpikes = 1;
% frameLength = 1000/30.9; % sec.
[C, S, C_df] = processEftyOuts(C, S, C_df, Nnan_nanBeg_nanEnd, normalizeSpikes);

% set time constants (in ms) from P.gn
tau = nan(size(P.gn,1), 2);
for i = 1:length(tau)
    g = P.gn{i};
    tau(i,:) = tau_d2c(g,frameLength); % tau(:,1) is rise, and tau(:,2) is decay time constant (in ms).
end
figure; histogram(tau(:,2))


%% Evaluate A and C of Efty's algorithm

if evaluateEftyAC
%     figure; imagesc(reshape(mean(P.psdx,2), imHeight, imWidth))

    load(imfilename, 'imHeight', 'imWidth', 'medImage')
    
    figure; imagesc(reshape(mean(A,2), imHeight, imWidth)) % look at average of spatial components.
    
    im = medImage{2};
    im = im - min(im(:)); softImMax = quantile(im(:), 0.995); im = im / softImMax; im(im > 1) = 1; % matt's method.
    
    contour_threshold = .95;
    plotCOMs = 1;
    
    [CC, ~, COMs] = setCC_cleanCC_plotCC_setMask(A, imHeight, imWidth, contour_threshold, im, plotCOMs);

    inds2plot = 1:size(C,1); % excl'; %excl(randperm(length(excl)))'; % size(C,1):-1:1; % 
    if ~exist('dFOF_man','var') % use this if you don't have manual activity
        plotEftManTracesROIs(C_df, S, [], A, [], CC, [], [], im, C, inds2plot, 0, 0, medImage{1});
    else % use this if you wan to compare with manual activity:
        plotEftManTracesROIs(C_df, S, dFOF_man', A, [], CC, [], 1:size(C_df,1), im, C, inds2plot, 0, 0, medImage{1});
        % traceQualManual = plotEftManTracesROIs(C_df, S_df, dFOF, A2, mask_eft, CC, CC_rois, eftMatchIdx_mask, im, C, inds2plot, manualTraceQual, plothists, im2)
    end
end


%% Evaluate C,f,manual activity, also tau, sn as well as some params related to A

% plot C, f, manual activity
figure; h = [];
subplot(411), plot(nanmean(S)); title('S'), h = [h, gca];
subplot(412), plot(nanmean(C)); title('C'), h = [h, gca];
subplot(413), plot(f); title('f'), h = [h, gca];
if exist('activity_man_eftMask', 'var')
    subplot(414), plot(mean(activity_man_eftMask, 2)), title('manual'), h = [h, gca];
else
    warning('activity_man_eftMask does not exist!')
end
linkaxes(h, 'x')

% Assess tau and noise for each neuron
% load(pnevFileName, 'P')
figure;
subplot(211), plot(tau(:,2)), xlabel('Neuron'), ylabel('Tau (ms)'), legend('decay') %, legend('rise','decay')
subplot(212), plot(cell2mat(P.neuron_sn)), xlabel('Neuron'), ylabel('neuron\_sn')

% Assess a number of parameters related to A (spatial component)
% load(pnevFileName, 'A')
figure;
subplot(311), plot(sum(A~=0,1)), ylabel('Number of pixels')
subplot(312), plot(nanmean(A,1)), ylabel('Mean A')
subplot(313), plot(max(A,[],1)), ylabel('Max A')
xlabel('Neuron')


%%
spikes = S';
activity = C'; %temporalComp'; % frames x units % temporalComp = C; % C_mcmc; % C2; % obj.C;
dFOF = C_df'; % temporalDf'; temporalDf = C_df; % C_mcmc_df; % C_df; % obj.C_df;

clear C C_df S



%%
%%%%%%%%%%%%%%%%%%%%%%%%% Take care of alldata %%%%%%%%%%%%%%%%%%%%%%%

%% Merge imaging variables into all_data (before removing any trials): activity, dFOF, spikes

minPts = 7000; %800;
[all_data, mscanLag] = mergeActivityIntoAlldata_fn(all_data, activity, framesPerTrial, ...
    trialNumbers, frame1RelToStartOff, badFrames{signalCh}, pmtOffFrames{signalCh}, minPts, dFOF, spikes);

% manual
% activity = activity_man;
% dFOF = dFOF_man;
% spikes = dFOF_man;
% clear dFOF_man activity_man
% [all_data, mscanLag] = mergeActivityIntoAlldata_fn(all_data, activity, framesPerTrial, ...
%   trialNumbers, frame1RelToStartOff, badFrames{signalCh}, pmtOffFrames{signalCh}, minPts, dFOF, spikes);

%{
%% Take care of helped trials: you don't need it. you've done it for all ur behavioral data and appended the helped fields to alldata and saved it. if not the following function will do it.
% [alldata_fileNames, ~] = setBehavFileNames(mouse, {datestr(datenum(imagingFolder, 'yymmdd'))});
% [all_data, ~] = loadBehavData(alldata_fileNames(mdfFileNumber), defaultHelpedTrs, saveHelpedTrs);

% If defaultHelpedTrs is false and alldata doesn't include the helped fields, add helpedInit and helpedChoice fields to alldata, and save it.

alldata_fileNam = imfilename;
all_data = setHelpedTrs(all_data, defaultHelpedTrs, saveHelpedTrs, alldata_fileNam, helpedInit, helpedChoice);
%}


%% Take care of trials that are in alldata but not in imaging data.

%{
if sum(~ismember(1:length(trialNumbers), trialNumbers))~=0
    error('There are non-triggered trials in mscan! Decide how to proceed!')
    % try this:
    % remove the trial that was not recorded in MScan
    all_data(~ismember(1:length(trialNumbers), trialNumbers)) = []; % FN note: double check this.
end
alldata = all_data(trialNumbers); % in case mscan crashed, you want to remove trials that were recorded in bcontrol but not in mscan.
%}

alldata = all_data;

% clear all_data
% alldata = alldata(1:end-1);  % you commented it here and added it above. loadBehavData by default removes the last trial.   % alldata = removeBegEndTrs(alldata, thbeg);
% fprintf('Total number of imaged trials: %d\n', length(alldata))


% Add NaN for alldata.dFOF, spikes, and activity of those trials that were
% not imaged (either at the end, or at the middle due to mscan failure.)

% For trials that were not imaged, set frameTimes, dFOF, spikes and activity traces to all nans (size:
% min(framesPerTrial) x #neurons).
[alldata([alldata.hasActivity]==0).dFOF] = deal(NaN(min(framesPerTrial), size(dFOF,2)));
[alldata([alldata.hasActivity]==0).spikes] = deal(NaN(min(framesPerTrial), size(dFOF,2)));
[alldata([alldata.hasActivity]==0).activity] = deal(NaN(min(framesPerTrial), size(dFOF,2)));
[alldata([alldata.hasActivity]==0).frameTimes] = deal(NaN(1, min(framesPerTrial)));


% imaging.alldata = alldata;


%% Set trs2rmv, stimrate, outcome and response side. You will set certain variables to NaN for trs2rmv (but you will never remove them from any arrays).

load(imfilename, 'badAlignTrStartCode', 'trialStartMissing'); %, 'trialCodeMissing') % they get set in framesPerTrialStopStart3An_fn

imagingFlg = 1;
[trs2rmv, stimdur, stimrate, stimtype, cb] = setTrs2rmv_final(alldata, thbeg, excludeExtraStim, excludeShortWaitDur, begTrs, imagingFlg, badAlignTrStartCode, trialStartMissing, trialCodeMissing);


%%%%% Set outcome and response side for each trial, taking into account allcorrection and uncommitted responses.
% Set some params related to behavior % behavior_info
[outcomes, allResp, allResp_HR_LR] = set_outcomes_allResp(alldata, uncommittedResp, allowCorrectResp);

% set trs2rmv to nan
outcomes(trs2rmv) = NaN;
allResp(trs2rmv) = NaN;
allResp_HR_LR(trs2rmv) = NaN;
% stimrate(trs2rmv) = NaN;

% save('151102_001.mat', '-append', 'trs2rmv')  % Do this!

% imaging.trs2rmv = trs2rmv;
% save(imfilename, '-append', 'imaging')


%% Set event times (ms) relative to when bcontrol starts sending the scope TTL. event times will be set to NaN for trs2rmv.

scopeTTLOrigTime = 1;
stimAftGoToneParams = {rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, setNaN_goToneEarlierThanStimOffset};
% stimAftGoToneParams = []; % {0,0,0};
[timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
    time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
    setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv, scopeTTLOrigTime, stimAftGoToneParams);

% alldata_frameTimes = {alldata.frameTimes};
% save(imfilename, '-append', 'alldata_frameTimes', 'timeStop')



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Take care of neural traces %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Assess trace quality of each neuron

if manualExamineTraceQual
    %     assessCaTraceQaulity % another script to look at traces 1 by 1.
    
    inds2plot = randperm(size(activity,2));
    traceQuality = plotEftManTracesROIs(dFOF', spikes', [], [], [], [], [], [], [], activity', inds2plot, 1, 0, []);
    
    if saveTraceQual
        save(imfilename, 'traceQuality', '-append')
    end
    goodNeurons = ismember(traceQuality, analyzeQuality);
end


%%%%% Automatic assessment of trace quality

% Compute measures of trace quality
if autoTraceQual
    [avePks2sdS, aveProm2sdS, measQual] = traceQualMeasure(dFOF', spikes');
    
    % badQual = find(avePks2sdS<3 | aveProm2sdS<1);
    badQual = find(measQual<0);
    
    goodNeurons = true(1,length(avePks2sdS));
    goodNeurons(badQual) = false;
    
    % look at C of neurons identified as badQual.
    %     plotCaTracesPerNeuron({dFOF(:, badQual)}, [], [], 0, 0, 0, 0)
    
    figure;
    for i = badQual, % 1:size(dFOF)
        plot(dFOF(:,i))
        %         plot(spikes(:,i))
        
        set(gcf,'name',num2str(i))
        title(sprintf('pks %.2f   prom %.2f   prod %.2f   meas %.2f', avePks2sdS(i), aveProm2sdS(i), ...
            avePks2sdS(i) * aveProm2sdS(i), measQual(i)))
        
        pause
    end
    
    if orderTraces
        % Sort traces
        [sPks, iPks] = sort(avePks2sdS);
        [sProm, iProm] = sort(aveProm2sdS);
        [sPksProm, iPksProm] = sort(avePks2sdS .* aveProm2sdS);
        [sMeasQual, iMeasQual] = sort(measQual);
        
        % order based on automatic measure
        goodnessOrder = iMeasQual(sMeasQual>=0); % ascending order of neurons based on trace quality
        goodnessOrder = goodnessOrder(end:-1:1); % descending order
    end
end
% fprintf('N good-quality and all neurons: %d, %d. Ratio: %.2f\n', [sum(goodNeurons), size(activity,2), sum(goodNeurons)/ size(activity,2)]); % number and fraction of good neurons vs number of all neurons.


% set goodinds: an array of length of all neurons, with 1s indicating good and 0s bad neurons.
if ~any([autoTraceQual, manualExamineTraceQual])  % if no trace quality examination (auto or manual), then include all neurons.
    goodinds = true(1, size(dFOF,2)); %    goodnessOrder = 1:size(dFOF,2);    goodNeurons = 1:size(dFOF,2);
elseif orderTraces % descending based on the measure of quality
    error('the code below needs work. goodinds is supposed to be logical not tr numbers.')
    goodinds = goodnessOrder;
else % stick to the original order
    goodinds = goodNeurons;
end

fprintf('N good-quality and all neurons: %d, %d. Ratio: %.2f\n', [sum(goodinds), size(activity,2), sum(goodinds)/ size(activity,2)]); % number and fraction of good neurons vs number of all neurons.



%% In alldata, set good quality traces.

% alldataDfofGood = cellfun(@(x)x(:, goodinds), {alldata.activity}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.
alldataDfofGood = cellfun(@(x)x(:, goodinds), {alldata.dFOF}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.
alldataSpikesGood = cellfun(@(x)x(:, goodinds), {alldata.spikes}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.

activityGood = activity(:, goodinds); % frames x units % remember activity and dFOF may have more frames that alldataDfofGood_mat bc alldataDfofGood_mat does not include the frames of the trial during which mscan was stopped but activity includes those frames.
dfofGood = dFOF(:, goodinds); % frames x units
spikesGood = spikes(:, goodinds); % frames x units

alldataDfofGood_mat = cell2mat(alldataDfofGood'); % frames x units;  % same as dfofGood except for some frames at the end that it may miss due to stopping mscan at the middle of a trial.

if compareManual
    eftMatchIdx_mask_good = matchedROI_idx(goodinds);
end

% alldataDfof = {alldata.dFOF};
% alldataDfof_mat = cell2mat(alldataDfof'); % frames x units


% numUnits = size(alldata(1).dFOF,2); % all units, regardless of the quality.
% numGoodUnits = size(dfofGood,2);
% numTrials = length(alldata);


%% Set inhibit and excit traces.

% alldataDfofGoodInh, alldataSpikesGoodInh and alldataDfofGoodExc, alldataSpikesGoodExc.
% size: 1 x number of trials, each element includes the traces of all inh (exc)
% neurons.


if setInhibitExcit
    
    % Load inhibitRois if it already exists, otherwise set it.
    
    a = matfile(imfilename);
    
    if isprop(a, 'inhibitRois')
        fprintf('Loading inhibitRois...\n')
        load(imfilename, 'inhibitRois')
        
    else
        fprintf('Identifying inhibitory neurons....\n')
        % inhibitRois will be :
        % 1 for inhibit ROIs.
        % 0 for excit ROIs.
        % nan for ROIs that could not be classified as inhibit or excit.
        [inhibitRois, roi2surr_sig, sigTh] = inhibit_excit_setVars(imfilename, pnevFileName, sigTh, assessInhibitClass);
        
        if saveInhibitRois
            fprintf('Saving inhibitRois...\n')
            save(imfilename, '-append', 'inhibitRois', 'roi2surr_sig', 'sigTh')
        end
        
    end
    
    fprintf('Fract inhibit %.3f, excit %.3f, unknown %.3f\n', [...
        nanmean(inhibitRois==1), nanmean(inhibitRois==0), nanmean(isnan(inhibitRois))])
    
    
    %% Set good_inhibit and good_excit neurons (ie in good quality neurons which ones are inhibit and which ones are excit).
    
    % goodinds: an array of length of all neurons, with 1s indicating good and 0s bad neurons.
    good_inhibit = inhibitRois(goodinds) == 1; % an array of length of good neurons, with 1s for inhibit. and 0s for excit. neurons and nans for unsure neurons.
    good_excit = inhibitRois(goodinds) == 0; % an array of length of good neurons, with 1s for excit. and 0s for inhibit neurons and nans for unsure neurons.
    
    fprintf('Fract inhibit %.3f, excit %.3f in good quality neurons\n', [...
        nanmean(good_inhibit), nanmean(good_excit)])
    
    % you can use the codes below if you want to be safe:
    %     good_inhibit = inhibitRois(goodinds & roi2surr_sig >= 1.3) == 1;
    %     good_excit = inhibitRois(goodinds & roi2surr_sig <= 1.1) == 0;
    
    if nanmean(goodinds) ~= 1
        fprintf('Fract inhibit in all, good & bad Ns = %.3f  %.3f  %.3f\n', [...
            nanmean(inhibitRois==1)
            nanmean(inhibitRois(goodinds)==1)
            nanmean(inhibitRois(~goodinds)==1)])
        % it seems good neurons include more excit (than inhibit) neurons
        % as if for some reason tdtomato neruons have low quality on the
        % green channel.
    end
    
    
    %% Set traces for good inhibit and excit neurons.
    
    alldataDfofGoodInh = cellfun(@(x)x(:, good_inhibit==1), alldataDfofGood, 'uniformoutput', 0); % 1 x number of trials
    alldataSpikesGoodInh = cellfun(@(x)x(:, good_inhibit==1), alldataSpikesGood, 'uniformoutput', 0); % 1 x number of trials
    
    alldataDfofGoodExc = cellfun(@(x)x(:, good_excit==1), alldataDfofGood, 'uniformoutput', 0); % 1 x number of trials
    alldataSpikesGoodExc = cellfun(@(x)x(:, good_excit==1), alldataSpikesGood, 'uniformoutput', 0); % 1 x number of trials
    
    % For the matrices below just do (:, good_inhibit) and (:, good_excit)
    % to get their corresponding traces for inhibit and excit neurons :
    % activityGood
    % dfofGood
    % spikesGood
    % alldataDfofGood_mat
    
else
    good_excit = [];
    good_inhibit = [];
end





%%
%%%%%%%%%%%%%%%%%%%%%%%%% Do some plotting of neural traces %%%%%%%%%%%%%%%%%%%%%%%

%% Look at traces per neuron and per trial and show events during a trial as well. Traces start at the scan start (so not aligned on any particular event.)

if plotTrialTraces1by1
    
    % it makes sense to look at all trials and not just ~trs2rmv so do
    % trs2rmv = unique([trs_problemAlign, trs_badMotion_pmtOff]);, and again run setEvent
    plotTrs1by1 = 1; %1;
    interactZoom = 0;
    markQuantPeaks = 1; % 1;
    showitiFrNums = 1;
    allEventTimes = {timeInitTone, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, timeStop, centerLicks, leftLicks, rightLicks};
    %     [~, ~, stimrate] = setStimRateType(alldata);
    
    % Remember the first input will be plotted!
    % If you want to plot certain neurons, specify that in dfofGood, eg: dFOF(:, badQual)
    plotCaTracesPerNeuron({dfofGood}, alldataDfofGood, alldataSpikesGood, interactZoom, plotTrs1by1, markQuantPeaks, showitiFrNums, {framesPerTrial, alldata, spikesGood, [], [], allEventTimes, stimrate})
    
    % plotCaTracesPerNeuron([{C_df'}, {C_mcmc_df'}], interactZoom, plotTrs1by1, markQuantPeaks, alldataDfofGood, alldataSpikesGood, {framesPerTrial, alldata, S_mcmc', dFOF_man, matchedROI_idx, allEventTimes, stimrate})
    % plot badQual neurons.
    %     plotCaTracesPerNeuron({dFOF(:, badQual)}, [], [], 0, 0, 0, 0)
    
    
    %% plot all neuron traces per trial
    
    plotCaTracesPerTrial
    
    
end


%% Plot average traces across all neurons and all trials aligned on particular trial events.

if plot_ave_noTrGroup

    outcome2ana = 'all'; % 'all'; 1: success, 0: failure, -1: early decision, -2: no decision, -3: wrong initiation, -4: no center commit, -5: no side commit
    stimrate2ana = 'all'; % 'all'; 'HR'; 'LR';
    strength2ana = 'all'; % 'all'; 'easy'; 'medium'; 'hard';    

    %%%%%% plot average imaging traces aligned on licks
    evT = {'centerLicks', 'leftLicks', 'rightLicks'}; % times are relative to scopeTTL onset, hence negative values are licks that happened before that (during iti states).
    nPreFrames = 5;
    nPostFrames = 20;    
    excludeLicksPrePost = 'none'; % 'none'; 'pre'; 'post'; 'both';
    
    avetrialAlign_plotAve_noTrGroup_licks(evT, outcome2ana, stimrate2ana, strength2ana, outcomes, stimrate, cb, alldata, alldataDfofGood, alldataSpikesGood, frameLength, nPreFrames, nPostFrames, centerLicks, leftLicks, rightLicks, excludeLicksPrePost)

    
    %%%%%% plot average imaging traces aligned on trial events
    evT = {'1', 'timeInitTone', 'timeStimOnset', 'timeStimOffset', 'timeCommitCL_CR_Gotone',...
        'time1stSideTry', 'time1stCorrectTry', 'time1stIncorrectTry',...
        'timeReward', 'timeCommitIncorrResp', 'timeStop'};
    
    avetrialAlign_plotAve_noTrGroup(evT, outcome2ana, stimrate2ana, strength2ana, trs2rmv, outcomes, stimrate, cb, alldata, alldataDfofGood, alldataSpikesGood, frameLength, timeInitTone, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, time1stIncorrectTry, timeReward, timeCommitIncorrResp, timeStop)
    
    
    %%%%%% plot average lick traces aligned on trial events.
    lickInds = [1, 2,3]; % Licks to analyze % 1: center lick, 2: left lick; 3: right lick
    
    lickAlign(lickInds, evT, outcome2ana, stimrate2ana, strength2ana, trs2rmv, outcomes, stimrate, cb, alldata, frameLength)
    
end


%% Plot average traces across all neurons for different trial groups aligned on particular trial events.

if plot_ave_trGroup
    fprintf('Remember: in each subplot (ie each neuron) different trials are contributing to different segments (alignments) of the trace!\n')
    avetrialAlign_plotAve_trGroup
end





%%
%%%%%%%%%%%%%%%%%%%%%%%%% Start some analyses: alignement, choice preference, SVM %%%%%%%%%%%%%%%%%%%%%%%

if furtherAnalyses
    %% Align traces on particular trial events
    
    % remember traces_al_sm has nan for trs2rmv as well as trs in alignedEvent that are nan.
    
    traces = alldataSpikesGood; % alldataSpikesGoodExc; % alldataSpikesGoodInh; % alldataSpikesGood;  % traces to be aligned.
    alignedEvent = 'stimOn'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward'
    dofilter = false; true; % false;
    % set nPre and nPost to nan if you want to go with the numbers that are based on eventBef and eventAft.
    % set to [] to include all frames before and after the alignedEvent.
    nPreFrames = nan; []; % nan;
    nPostFrames = nan; []; % nan;
    
    traceTimeVec = {alldata.frameTimes}; % time vector of the trace that you want to realign.
    
    [traces_al_sm, time_aligned_stimOn, eventI_stimOn] = alignTraces_prePost_filt...
        (traces, traceTimeVec, alignedEvent, frameLength, dofilter, timeInitTone, timeStimOnset, ...
        timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, nPreFrames, nPostFrames);
    
    % set to nan those trials in outcomes and allRes that are nan in traces_al_sm
    a = find(sum(sum(~isnan(traces_al_sm),1),3), 1);
    allTrs2rmv = find(squeeze(sum(isnan(traces_al_sm(:,a,:)))));
    outcomes(allTrs2rmv) = NaN;
    allResp(allTrs2rmv) = NaN;
    allResp_HR_LR(allTrs2rmv) = NaN;
    
    
    %% Compute and plot choice preference, 2*(auc-0.5), for each neuron at each frame.
    
    % choicePref_ROC
    
    % set ipsi and contra trials
    thStimStrength = 2; % 2; % what stim strength you want to use for computing choice pref.
    
    correctL = (outcomes==1) & (allResp==1);
    correctR = (outcomes==1) & (allResp==2);
    
    ipsiTrs = (correctL' &  abs(stimrate-cb) > thStimStrength);
    contraTrs = (correctR' &  abs(stimrate-cb) > thStimStrength);
    fprintf('Num corrL and corrR (stim diff > %d): %d,  %d\n', [thStimStrength, sum(ipsiTrs) sum(contraTrs)])
    
    makeplots = 1;
    useEqualNumTrs = false; % if true, equal number of trials for HR and LR will be used to compute ROC.
    
    % compute choicePref for each frame
    % choicePref_all: frames x units. chiocePref at each frame for each neuron
    choicePref_all = choicePref_ROC(traces_al_sm, ipsiTrs, contraTrs, makeplots, eventI_stimOn, useEqualNumTrs);
    
    %%% Compute choicePref for the average of frames after the stim.
    % traces_al_sm_aveFr = nanmean(traces_al_sm(eventI_stimOn:end,:,:), 1);
    % choicePref_all = choicePref_ROC(traces_al_sm_aveFr, ipsiTrs, contraTrs, makeplots, eventI_stimOn, useEqualNumTrs);
    
    
    %% SVM
    
    popClassifier
    
    
end


%%
%% Set problematic trials to later exclude them if desired.

% commenting for now but you may need it later.
%{
% in set_outcome_allResp you can take care of the following terms... u
% don't need to worry about them here.
% Trials that the mouse entered the allow correction state.
a = arrayfun(@(x)x.parsedEvents.states.punish_allowcorrection, alldata, 'uniformoutput', 0);
trs_allowCorrectEntered = find(~cellfun(@isempty, a));

% Trials that mouse licked the error side during the decision time. Afterwards, the mouse may have committed it
% (hence entered allow correction) or may have licked the correct side.
a = arrayfun(@(x)x.parsedEvents.states.errorlick_again_wait, alldata, 'uniformoutput', 0);
trs_errorlick_again_wait_entered = find(~cellfun(@isempty, a));


% Trials with unwanted outcomes.
% suc*, fail*, early*, no dec*/cho*, wrong st*/in*, no cen* com*, no s* com
% labels = [1 0 -1 -2 -3 -4 -5];
if strcmp(analyzeOutcomes, 'all')
    trs_unwantedOutcome = [];
else
    [~, trs_unwantedOutcome] = trialsToAnalyze(alldata, analyzeOutcomes);
end


% Trials during which mouse was running.
trs_mouseRunning = [];

%{
thWheel = .05;
wheelRevolution = {alldata.wheelRev};
rangeWheelRev = cellfun(@range, wheelRevolution);

max(rangeWheelRev)
min(rangeWheelRev)
mean(rangeWheelRev)

trs_mouseRunning = find(rangeWheelRev > thWheel); % trials in which mouse moved more than thWheel
length(trs_mouseRunning)
% trs2rmv = [];
%}

%}